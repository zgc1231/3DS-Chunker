from pathlib import Path
import json
import logging
import re
import shutil
import struct
import zlib

from anvil import Region, Block

from .convert import parse_block_json

logger = logging.getLogger(__name__)

OVERWORLD = 0
NETHER = 1
END = 2

MAGIC_CDB = 0xABCDEF98
FILE_HEADER_SIZE = 0x14
CHUNK_HEADER_SIZE = 0x6C
DEFAULT_SUBFILE_SIZE = 0x2800


def pos_pack(x: int, z: int, dim: int) -> int:
    return (x & 0x3FFF) | ((z & 0x3FFF) << 14) | ((dim & 0xF) << 28)


def block_key(block: Block) -> tuple[str, str, tuple[tuple[str, str], ...]]:
    properties = tuple(sorted((block.properties or {}).items()))
    return block.namespace, block.id, properties


def build_inverse_block_map(blocks: dict[tuple[int, int], Block]) -> dict:
    inverse = {}
    for block_id, block in blocks.items():
        inverse[block_key(block)] = block_id
    return inverse


def align(value: int, multiple: int = 0x10) -> int:
    return (value + multiple - 1) // multiple * multiple


def pack_chunk_header(
    chunk_x: int,
    chunk_z: int,
    dimension: int,
    chunk_status: tuple[int, int],
    section_offset: int,
    compressed_size: int,
    decompressed_size: int,
) -> bytes:
    header = bytearray()
    header.extend(
        struct.pack(
            "<IbbHHH",
            pos_pack(chunk_x, chunk_z, dimension),
            chunk_status[0],
            chunk_status[1],
            0,
            3,
            0,
        )
    )
    # first section contains the block data blob
    header.extend(
        struct.pack(
            "<iiii",
            0,
            section_offset,
            compressed_size,
            decompressed_size,
        )
    )
    # remaining unused sections
    for _ in range(5):
        header.extend(struct.pack("<iiii", -1, -1, 0, 0))
    return bytes(header)


def pack_cdb_entry(
    chunk_x: int,
    chunk_z: int,
    dimension: int,
    slot: int,
    subfile: int,
    chunk_status: tuple[int, int],
) -> bytes:
    return struct.pack(
        "<IHHHHbbH",
        pos_pack(chunk_x, chunk_z, dimension),
        slot,
        subfile,
        0x20FF,
        0x000A,
        chunk_status[0],
        chunk_status[1],
        0x8000,
    )


def pack_file_header(subfile_count: int, subfile_size: int) -> bytes:
    return struct.pack("<HHIIII", 1, 1, subfile_count, FILE_HEADER_SIZE, subfile_size, 0x4)


def pack_index(
    entries: list[bytes],
    pointer_count: int = 1,
    unknown0: int = 0x3E04,
) -> bytes:
    header = struct.pack(
        "<IIIIII",
        0x2,
        len(entries),
        unknown0,
        0x10,
        pointer_count,
        0x80,
    )
    pointers = b"".join(struct.pack("<I", i) for i in range(pointer_count))
    return header + pointers + b"".join(entries)


def chunk_block_payload(java_chunk, inverse_block_map: dict) -> bytes:
    subchunks: list[bytes] = []
    highest_present = -1
    for subchunk_index in range(8):
        blocks = bytearray(16 * 16 * 16)
        block_data = bytearray(16 * 16 * 16 // 2)
        unknown_block_data = bytearray(16 * 16 * 16)
        has_blocks = False

        for x in range(16):
            for z in range(16):
                for y in range(16):
                    world_y = subchunk_index * 16 + y
                    block = java_chunk.get_block(x, world_y, z)
                    mapped = inverse_block_map.get(block_key(block))
                    if mapped is None:
                        mapped = (0, 0)
                    block_id, block_meta = mapped
                    index = x * 16 * 16 + z * 16 + y
                    blocks[index] = block_id
                    if index % 2 == 0:
                        block_data[index // 2] |= block_meta & 0xF
                    else:
                        block_data[index // 2] |= (block_meta & 0xF) << 4
                    if block_id or block_meta:
                        has_blocks = True

        if has_blocks:
            highest_present = subchunk_index

        payload = bytearray()
        payload.append(0)
        payload.extend(blocks)
        payload.extend(block_data)
        payload.extend(unknown_block_data)
        subchunks.append(bytes(payload))

    subchunk_count = highest_present + 1
    data = bytearray()
    data.append(subchunk_count)
    for subchunk_index in range(subchunk_count):
        data.extend(subchunks[subchunk_index])
    # heightmap + biomes placeholders
    data.extend(b"\0" * (16 * 16 * 2))
    data.extend(b"\0" * (16 * 16))
    return bytes(data)


class CdbBuilder:
    def __init__(self, output_directory: Path, chunk_status: tuple[int, int]) -> None:
        self.output_directory = Path(output_directory)
        self.chunk_status = chunk_status
        self.entries: list[bytes] = []
        self.chunk_blobs: list[tuple[bytes, int]] = []
        self.chunk_positions: list[tuple[int, int, int]] = []
        self.max_subfile_size = 0

    def add_chunk(self, chunk_x: int, chunk_z: int, dimension: int, payload: bytes) -> None:
        compressed = zlib.compress(payload)
        subfile_min_size = (
            len(compressed) + CHUNK_HEADER_SIZE + struct.calcsize("<I")
        )
        self.max_subfile_size = max(self.max_subfile_size, subfile_min_size)
        self.chunk_blobs.append((compressed, len(payload)))
        self.chunk_positions.append((chunk_x, chunk_z, dimension))
        subfile_index = len(self.chunk_blobs) - 1
        self.entries.append(
            pack_cdb_entry(
                chunk_x,
                chunk_z,
                dimension,
                0,
                subfile_index,
                self.chunk_status,
            )
        )

    def _write_slot(self, subfile_size: int) -> None:
        slot_path = self.output_directory / "slt0.cdb"
        slot_path.parent.mkdir(parents=True, exist_ok=True)
        with open(slot_path, "wb") as slot_file:
            slot_file.write(pack_file_header(len(self.chunk_blobs), subfile_size))
            for (compressed_blob, decompressed_size), position in zip(
                self.chunk_blobs, self.chunk_positions
            ):
                section_offset = len(struct.pack("<I", MAGIC_CDB)) + CHUNK_HEADER_SIZE
                chunk_header = pack_chunk_header(
                    position[0],
                    position[1],
                    position[2],
                    self.chunk_status,
                    section_offset,
                    len(compressed_blob),
                    decompressed_size,
                )
                subfile = bytearray()
                subfile.extend(struct.pack("<I", MAGIC_CDB))
                subfile.extend(chunk_header)
                subfile.extend(compressed_blob)
                padding = subfile_size - len(subfile)
                if padding < 0:
                    raise ValueError("subfile size too small for chunk data")
                subfile.extend(b"\0" * padding)
                slot_file.write(subfile)

    def _write_index(self) -> None:
        index_bytes = pack_index(self.entries)
        for index_name in ("index.cdb", "newindex.cdb"):
            with open(self.output_directory / index_name, "wb") as index_file:
                index_file.write(index_bytes)

    def write(self) -> None:
        if not self.chunk_blobs:
            logger.warning("No chunks to write to CDB output")
            return
        subfile_size = align(max(self.max_subfile_size, DEFAULT_SUBFILE_SIZE))
        self._write_slot(subfile_size)
        self._write_index()


def iter_java_chunks(java_world: Path):
    region_pattern = re.compile(r"r\.(-?\d+)\.(-?\d+)\.mca")
    for region_file in (java_world / "region").glob("r.*.*.mca"):
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
                chunk = region.get_chunk(local_x, local_z)
                yield (
                    chunk,
                    region_x * 32 + local_x,
                    region_z * 32 + local_z,
                    OVERWORLD,
                )


def convert_java(java_world: Path, world_out: Path, delete_out: bool) -> None:
    if world_out.exists():
        if delete_out:
            for child in world_out.iterdir():
                if child.is_file():
                    child.unlink()
                else:
                    shutil.rmtree(child)
        else:
            raise FileExistsError("output directory already exists, pass --delete-out to overwrite")
    world_out.mkdir(parents=True, exist_ok=True)

    with open(Path(__file__).parent / "data" / "blocks.json") as blocks_file:
        raw_blocks = json.load(blocks_file)
    blocks = parse_block_json(raw_blocks)
    inverse_block_map = build_inverse_block_map(blocks)

    cdb_output = world_out / "db" / "cdb"
    builder = CdbBuilder(cdb_output, (7, 0))

    for chunk, chunk_x, chunk_z, dimension in iter_java_chunks(java_world):
        payload = chunk_block_payload(chunk, inverse_block_map)
        builder.add_chunk(chunk_x, chunk_z, dimension, payload)

    builder.write()
