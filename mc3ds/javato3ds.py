from pathlib import Path
import json
import logging
import os
import re
import shutil
import struct
from concurrent.futures import ProcessPoolExecutor, as_completed
import zlib

from anvil import Region, Block

logger = logging.getLogger(__name__)

OVERWORLD = 0
NETHER = 1
END = 2

MAGIC_CDB = 0xABCDEF98
FILE_HEADER_SIZE = 0x14
SUBFILE_HEADER_SIZE = 0x4
CHUNK_HEADER_SIZE = 0x6C
DEFAULT_SUBFILE_SIZE = 0x2800
DEFAULT_POINTER_COUNT = 28
DEFAULT_INDEX_UNKNOWN0 = 0x3E04
MAX_SLOT_SIZE_KB = 1281
MAX_SLOT_SIZE_BYTES = MAX_SLOT_SIZE_KB * 1024

_INVERSE_BLOCK_MAP: dict | None = None

BLOCK_DEF_PATTERN = re.compile(r"^([^\[\]]+)(?:\[([^\[\]]*)\])?$")


def pos_pack(x: int, z: int, dim: int) -> int:
    return (x & 0x3FFF) | ((z & 0x3FFF) << 14) | ((dim & 0xF) << 28)


def block_key(block: Block) -> tuple[str, str, tuple[tuple[str, str], ...]]:
    block_id = getattr(block, "id", "")
    if not isinstance(block_id, str):
        block_id = str(block_id)
    if ":" in block_id:
        namespace, name = block_id.split(":", 1)
    else:
        namespace, name = "minecraft", block_id
    props = getattr(block, "properties", None)
    if not isinstance(props, dict):
        props_dict: dict[str, str] = {}
    else:
        props_dict = props
    properties = tuple(sorted(props_dict.items()))
    return namespace, name, properties


def parse_block_definition(definition: str) -> tuple[str, str, tuple[tuple[str, str], ...]]:
    matched = BLOCK_DEF_PATTERN.match(definition)
    if matched is None:
        raise ValueError(f"invalid block descriptor {definition!r}")
    name = matched[1]
    raw_props = matched[2]
    if ":" in name:
        namespace, block_name = name.split(":", 1)
    else:
        namespace, block_name = "minecraft", name
    if raw_props:
        props = dict(item.split("=", 1) for item in raw_props.split(","))
    else:
        props = {}
    return namespace, block_name, tuple(sorted(props.items()))


def build_inverse_block_map(raw_blocks: dict) -> dict:
    mapping: dict[tuple[str, str, tuple[tuple[str, str], ...]], tuple[int, int]] = {}
    for numerical_id, descriptor in raw_blocks.get("blocks", {}).items():
        namespace, name, props = parse_block_definition(descriptor)
        id_part, data_part = numerical_id.split(":", 1)
        mapping[(namespace, name, props)] = (int(id_part), int(data_part))
    return mapping


class CdbSlotBuilder:
    def __init__(
        self,
        slot_index: int,
        subfile_size: int = DEFAULT_SUBFILE_SIZE,
        header_size: int = FILE_HEADER_SIZE,
    ) -> None:
        self.slot_index = slot_index
        self.subfile_size = subfile_size
        self.header_size = header_size
        self.subfiles: list[bytes] = []

    def projected_size_with(self, additional_subfiles: int = 0) -> int:
        return self.header_size + (len(self.subfiles) + additional_subfiles) * self.subfile_size

    def add_chunk(
        self, position: int, parameters: tuple[int, int], compressed: bytes, decompressed: int
    ) -> int:
        data_offset = SUBFILE_HEADER_SIZE + CHUNK_HEADER_SIZE
        if data_offset + len(compressed) > self.subfile_size:
            raise ValueError("chunk payload does not fit into subfile")

        page = bytearray(self.subfile_size)
        struct.pack_into("<I", page, 0x0, MAGIC_CDB)
        offset = SUBFILE_HEADER_SIZE
        struct.pack_into(
            "<IbbHHH",
            page,
            offset,
            position,
            parameters[0],
            parameters[1],
            0,
            3,
            0,
        )
        offset += 12

        sections = [
            (0, data_offset, len(compressed), decompressed),
            (-1, -1, 0, 0),
            (-1, -1, 0, 0),
            (-1, -1, 0, 0),
            (-1, -1, 0, 0),
            (-1, -1, 0, 0),
        ]
        for index, position_offset, compressed_size, decompressed_size in sections:
            struct.pack_into(
                "<iiii", page, offset, index, position_offset, compressed_size, decompressed_size
            )
            offset += 16

        page[data_offset : data_offset + len(compressed)] = compressed
        self.subfiles.append(bytes(page))
        return len(self.subfiles) - 1

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            header = struct.pack(
                "<HHIIII",
                1,
                1,
                len(self.subfiles),
                self.header_size,
                self.subfile_size,
                0x4,
            )
            handle.write(header)
            if self.header_size > len(header):
                handle.write(b"\x00" * (self.header_size - len(header)))
            for subfile in self.subfiles:
                if len(subfile) < self.subfile_size:
                    handle.write(subfile)
                    handle.write(b"\x00" * (self.subfile_size - len(subfile)))
                else:
                    handle.write(subfile)


class CdbIndexBuilder:
    def __init__(self, pointer_count: int = DEFAULT_POINTER_COUNT) -> None:
        self.pointer_count = pointer_count
        self.entries: list[tuple[int, int, int, tuple[int, int]]] = []

    def add_entry(
        self, position: int, slot: int, subfile: int, parameters: tuple[int, int]
    ) -> None:
        self.entries.append((position, slot, subfile, parameters))

    def _final_pointer_count(self) -> int:
        if not self.entries:
            return max(self.pointer_count, 1)
        max_slot = max(entry[1] for entry in self.entries)
        return max(self.pointer_count, max_slot + 1)

    def write(self, path: Path) -> None:
        pointer_count = self._final_pointer_count()
        entry_count = len(self.entries)
        header = struct.pack(
            "<IIIIII", 2, entry_count, DEFAULT_INDEX_UNKNOWN0, 16, pointer_count, 0x80
        )
        pointers = b"".join(struct.pack("<I", i) for i in range(pointer_count))
        entries_blob = bytearray()
        for position, slot, subfile, parameters in self.entries:
            entries_blob.extend(
                struct.pack(
                    "<IHHHHbbH",
                    position,
                    slot,
                    subfile,
                    0x20FF,
                    0x000A,
                    parameters[0],
                    parameters[1],
                    0x8000,
                )
            )
        payload = header + pointers + entries_blob
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            handle.write(payload)


class CdbWorldBuilder:
    def __init__(
        self,
        subfile_size: int = DEFAULT_SUBFILE_SIZE,
        pointer_count: int = DEFAULT_POINTER_COUNT,
        chunk_parameters: tuple[int, int] = (7, 0),
        max_slot_size_bytes: int = MAX_SLOT_SIZE_BYTES,
    ) -> None:
        self.subfile_size = subfile_size
        self.chunk_parameters = chunk_parameters
        self.max_slot_size_bytes = max_slot_size_bytes
        self.slots: dict[int, CdbSlotBuilder] = {}
        self.index_builder = CdbIndexBuilder(pointer_count)

    def _current_slot(self) -> CdbSlotBuilder:
        if not self.slots:
            slot_index = 0
            self.slots[slot_index] = CdbSlotBuilder(slot_index, self.subfile_size)
        slot_index = max(self.slots)
        slot = self.slots[slot_index]
        if slot.projected_size_with(1) > self.max_slot_size_bytes:
            slot_index += 1
            slot = CdbSlotBuilder(slot_index, self.subfile_size)
            self.slots[slot_index] = slot
        return slot

    def add_chunk(self, x: int, z: int, dimension: int, payload: bytes) -> None:
        position = pos_pack(x, z, dimension)
        slot = self._current_slot()
        slot_index = slot.slot_index
        compressed = zlib.compress(payload)
        subfile_index = slot.add_chunk(position, self.chunk_parameters, compressed, len(payload))
        self.index_builder.add_entry(position, slot_index, subfile_index, self.chunk_parameters)

    def write(self, output_directory: Path) -> None:
        output_directory = Path(output_directory)
        cdb_dir = output_directory / "db" / "cdb"
        cdb_dir.mkdir(parents=True, exist_ok=True)
        for slot_index, slot in sorted(self.slots.items()):
            slot.write(cdb_dir / f"slt{slot_index}.cdb")
        index_path = cdb_dir / "index.cdb"
        new_index_path = cdb_dir / "newindex.cdb"
        self.index_builder.write(index_path)
        self.index_builder.write(new_index_path)


def chunk_block_payload(java_chunk, inverse_block_map: dict) -> bytes:
    subchunk_count = 8
    subchunk_height = 16
    width = 16
    depth = 16
    blocks = [bytearray(width * depth * subchunk_height) for _ in range(subchunk_count)]
    metas = [bytearray(width * depth * subchunk_height // 2) for _ in range(subchunk_count)]
    unknown_block_data = b"\x00" * (width * depth * subchunk_height)
    max_y = subchunk_count * subchunk_height
    for y in range(max_y):
        sub = y // subchunk_height
        local_y = y % subchunk_height
        for z in range(depth):
            for x in range(width):
                block = java_chunk.get_block(x, y, z)
                key = block_key(block)
                block_id, block_data = inverse_block_map.get(key, (0, 0))
                position = x * 16 * 16 + z * 16 + local_y
                blocks[sub][position] = block_id & 0xFF
                meta_index = position // 2
                if position % 2 == 0:
                    metas[sub][meta_index] = (metas[sub][meta_index] & 0xF0) | (block_data & 0x0F)
                else:
                    metas[sub][meta_index] = (metas[sub][meta_index] & 0x0F) | (
                        (block_data & 0x0F) << 4
                    )
    out = bytearray()
    out.append(subchunk_count)
    for sub in range(subchunk_count):
        out.append(0)
        out.extend(blocks[sub])
        out.extend(metas[sub])
        out.extend(unknown_block_data)
    return bytes(out)


def resolve_region_path(java_world: Path) -> Path:
    java_world = Path(java_world)
    candidates = [
        java_world / "region",
        java_world / "DIM0" / "region",
        java_world,
    ]
    for candidate in candidates:
        if candidate.is_dir() and list(candidate.glob("r.*.*.mca")):
            return candidate
    raise FileNotFoundError(f"could not find region directory under {java_world!r}")


def _init_chunk_worker(inverse_block_map: dict) -> None:
    global _INVERSE_BLOCK_MAP
    _INVERSE_BLOCK_MAP = inverse_block_map


def _convert_chunk(
    region_file: Path,
    region_x: int,
    region_z: int,
    local_x: int,
    local_z: int,
    dimension: int,
):
    with open(region_file, "rb") as region_handle:
        region = Region.from_file(region_handle)
    if region.chunk_location(local_x, local_z) == (0, 0):
        return None
    chunk = region.get_chunk(local_x, local_z)
    if _INVERSE_BLOCK_MAP is None:
        raise RuntimeError("inverse block map not initialized")
    payload = chunk_block_payload(chunk, _INVERSE_BLOCK_MAP)
    chunk_x = region_x * 32 + local_x
    chunk_z = region_z * 32 + local_z
    return chunk_x, chunk_z, dimension, payload


def iter_java_chunks(java_world: Path):
    region_pattern = re.compile(r"r\.(-?\d+)\.(-?\d+)\.mca")
    region_dir = resolve_region_path(java_world)
    for region_file in region_dir.glob("r.*.*.mca"):
        match = region_pattern.fullmatch(region_file.name)
        if match is None:
            continue
        region_x, region_z = int(match.group(1)), int(match.group(2))
        with open(region_file, "rb") as region_handle:
            region = Region.from_file(region_handle)
        for local_x in range(32):
            for local_z in range(32):
                if region.chunk_location(local_x, local_z) == (0, 0):
                    continue
                try:
                    region.get_chunk(local_x, local_z)
                except KeyError as exc:
                    msg = str(exc)
                    if "xPos" in msg or "Level" in msg:
                        logger.debug(
                            "Skipping chunk (%d, %d) in region (%d, %d) due to missing xPos/Level",
                            local_x,
                            local_z,
                            region_x,
                            region_z,
                        )
                        yield (
                            region_file,
                            region_x,
                            region_z,
                            local_x,
                            local_z,
                            OVERWORLD,
                            False,
                        )
                        continue
                    raise
                yield (
                    region_file,
                    region_x,
                    region_z,
                    local_x,
                    local_z,
                    OVERWORLD,
                    True,
                )


def convert_java(java_world: Path, world_out: Path, delete_out: bool) -> None:
    if world_out.exists():
        if delete_out:
            shutil.rmtree(world_out)
        else:
            raise FileExistsError(
                "output directory already exists, pass --delete-out to overwrite",
            )
    world_out.mkdir(parents=True, exist_ok=True)
    db_dir = world_out / "db"
    cdb_dir = db_dir / "cdb"
    vdb_dir = db_dir / "vdb"
    db_dir.mkdir(exist_ok=True)
    cdb_dir.mkdir(exist_ok=True)
    vdb_dir.mkdir(exist_ok=True)
    (db_dir / "savemarker").touch(exist_ok=True)
    for placeholder in (world_out / "level.dat", world_out / "level.dat_old"):
        if not placeholder.exists():
            placeholder.write_bytes(b"\x00" * 8)

    with open(Path(__file__).parent / "data" / "blocks.json") as blocks_file:
        raw_blocks = json.load(blocks_file)
    inverse_block_map = build_inverse_block_map(raw_blocks)
    builder = CdbWorldBuilder()
    converted_chunks = 0
    missing_xpos_chunks = 0
    max_workers = max(1, (os.cpu_count() or 1) - 1)
    logger.info("Using %d worker process(es) for Java chunk conversion", max_workers)
    if max_workers == 1:
        _init_chunk_worker(inverse_block_map)
        for (
            region_file,
            region_x,
            region_z,
            local_x,
            local_z,
            dimension,
            has_xpos,
        ) in iter_java_chunks(java_world):
            if not has_xpos:
                missing_xpos_chunks += 1
                continue
            result = _convert_chunk(
                region_file,
                region_x,
                region_z,
                local_x,
                local_z,
                dimension,
            )
            if result is None:
                continue
            chunk_x, chunk_z, dimension, payload = result
            builder.add_chunk(chunk_x, chunk_z, dimension, payload)
            converted_chunks += 1
    else:
        max_in_flight = max_workers * 4
        futures = set()
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_init_chunk_worker,
            initargs=(inverse_block_map,),
        ) as executor:
            for (
                region_file,
                region_x,
                region_z,
                local_x,
                local_z,
                dimension,
                has_xpos,
            ) in iter_java_chunks(java_world):
                if not has_xpos:
                    missing_xpos_chunks += 1
                    continue
                fut = executor.submit(
                    _convert_chunk,
                    region_file,
                    region_x,
                    region_z,
                    local_x,
                    local_z,
                    dimension,
                )
                futures.add(fut)
                if len(futures) >= max_in_flight:
                    done = next(as_completed(futures))
                    futures.remove(done)
                    result = done.result()
                    if result is None:
                        continue
                    chunk_x, chunk_z, dimension, payload = result
                    builder.add_chunk(chunk_x, chunk_z, dimension, payload)
                    converted_chunks += 1
            for done in as_completed(futures):
                result = done.result()
                if result is None:
                    continue
                chunk_x, chunk_z, dimension, payload = result
                builder.add_chunk(chunk_x, chunk_z, dimension, payload)
                converted_chunks += 1
    builder.write(world_out)
    if converted_chunks == 0 and missing_xpos_chunks > 0:
        logger.error(
            "Conversion failed: all %d chunks were skipped because they were missing an xPos tag",
            missing_xpos_chunks,
        )
    logger.info("Conversion complete: %d chunks converted", converted_chunks)
