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
    block_id = getattr(block, "id", "")
    if ":" in block_id:
        namespace, name = block_id.split(":", 1)
    else:
        namespace, name = "minecraft", block_id
    props = getattr(block, "properties", None)
    if not isinstance(props, dict):
        props_dict = {}
    else:
        props_dict = props
    properties = tuple(sorted(props_dict.items()))
    return namespace, name, properties


def build_inverse_block_map(blocks: dict) -> dict:
    inverse: dict[tuple[str, str, tuple[tuple[str, str], ...]], int] = {}
    for _id, block_def in blocks.items():
        if isinstance(block_def, dict):
            namespace = block_def.get("namespace")
            name = block_def.get("name")
            variants = block_def.get("variants", [])
        else:
            namespace = getattr(block_def, "namespace", None)
            name = getattr(block_def, "name", None)
            variants = getattr(block_def, "variants", [])
        if not namespace or not name:
            continue
        for variant in variants:
            if isinstance(variant, dict):
                props = variant.get("properties", {})
                var_id = variant.get("id")
            else:
                props = getattr(variant, "properties", {}) or {}
                var_id = getattr(variant, "id", None)
            if var_id is None:
                continue
            key = (namespace, name, tuple(sorted(props.items())))
            if key in inverse:
                continue
            inverse[key] = var_id
    return inverse


class CdbBuilder:
    def __init__(self, output_directory: Path, chunk_status: tuple[int, int]) -> None:
        self.output_directory = Path(output_directory)
        self.chunk_status = chunk_status
        self.entries: list[bytes] = []
        self.chunk_blobs: list[tuple[bytes, int]] = []
        self._index_by_position: dict[int, int] = {}

    def add_chunk(self, x: int, z: int, dimension: int, payload: bytes) -> None:
        pos = pos_pack(x, z, dimension)
        if pos in self._index_by_position:
            raise ValueError(f"chunk at position {x}, {z}, {dimension} already added")
        index = len(self.entries)
        self._index_by_position[pos] = index
        header = bytearray(CHUNK_HEADER_SIZE)
        struct.pack_into("<I", header, 0x0, pos)
        struct.pack_into("<I", header, 0x4, 0)
        struct.pack_into("<I", header, 0x8, 0)
        struct.pack_into("<I", header, 0xC, 0)
        struct.pack_into("<I", header, 0x10, 0)
        struct.pack_into("<I", header, 0x14, 0)
        struct.pack_into("<I", header, 0x18, 0)
        struct.pack_into("<I", header, 0x1C, 0)
        struct.pack_into("<I", header, 0x20, 0)
        struct.pack_into("<I", header, 0x24, 0)
        struct.pack_into("<I", header, 0x28, 0)
        struct.pack_into("<I", header, 0x2C, 0)
        struct.pack_into("<I", header, 0x30, 3)
        struct.pack_into("<I", header, 0x34, self.chunk_status[0])
        struct.pack_into("<I", header, 0x38, self.chunk_status[1])
        struct.pack_into("<I", header, 0x3C, 0)
        struct.pack_into("<I", header, 0x40, 0)
        struct.pack_into("<I", header, 0x44, 0)
        struct.pack_into("<I", header, 0x48, 0)
        struct.pack_into("<I", header, 0x4C, 0)
        struct.pack_into("<I", header, 0x50, 0)
        struct.pack_into("<I", header, 0x54, 0)
        struct.pack_into("<I", header, 0x58, 0)
        struct.pack_into("<I", header, 0x5C, 0)
        struct.pack_into("<I", header, 0x60, 0)
        struct.pack_into("<I", header, 0x64, 0)
        struct.pack_into("<I", header, 0x68, 0)
        compressed = zlib.compress(payload)
        self.entries.append(bytes(header))
        self.chunk_blobs.append((compressed, len(payload)))

    def write(self) -> None:
        self.output_directory.mkdir(parents=True, exist_ok=True)
        output_path = self.output_directory / "0.cdb"
        with open(output_path, "wb") as out:
            self._write_file(out)

    def _write_file(self, out) -> None:
        entries_offset = FILE_HEADER_SIZE
        chunks_offset = FILE_HEADER_SIZE + CHUNK_HEADER_SIZE * len(self.entries)
        out.seek(0)
        out.write(b"\x00" * FILE_HEADER_SIZE)
        out.seek(entries_offset)
        for entry in self.entries:
            out.write(entry)
        out.seek(chunks_offset)
        chunk_offsets: list[tuple[int, int, int]] = []
        for blob, uncompressed_size in self.chunk_blobs:
            offset = out.tell()
            out.write(blob)
            size = out.tell() - offset
            if size > 0xFFFFFFFF or offset > 0xFFFFFFFF:
                raise ValueError("chunk data too large to write to cdb")
            chunk_offsets.append((offset, size, uncompressed_size))
        out.seek(0, os.SEEK_END)
        file_size = out.tell()
        header = bytearray(FILE_HEADER_SIZE)
        struct.pack_into("<I", header, 0x0, MAGIC_CDB)
        struct.pack_into("<I", header, 0x4, file_size)
        struct.pack_into("<I", header, 0x8, 1)
        struct.pack_into("<I", header, 0xC, 1)
        struct.pack_into("<I", header, 0x10, len(self.entries))
        out.seek(0)
        out.write(header)
        out.seek(FILE_HEADER_SIZE)
        for i, entry in enumerate(self.entries):
            offset, size, uncompressed_size = chunk_offsets[i]
            patched = bytearray(entry)
            struct.pack_into("<I", patched, 0x4, offset)
            struct.pack_into("<I", patched, 0x8, size)
            struct.pack_into("<I", patched, 0xC, uncompressed_size)
            out.write(patched)


def chunk_block_payload(java_chunk, inverse_block_map: dict) -> bytes:
    subchunk_count = 8
    subchunk_height = 16
    width = 16
    depth = 16
    subchunk_blocks = [bytearray(width * depth * subchunk_height) for _ in range(subchunk_count)]
    max_y = subchunk_count * subchunk_height
    for y in range(max_y):
        sub = y // subchunk_height
        local_y = y % subchunk_height
        for z in range(depth):
            for x in range(width):
                block = java_chunk.get_block(x, y, z)
                idx = (local_y * depth + z) * width + x
                key = block_key(block)
                block_id = inverse_block_map.get(key, 0)
                block_id &= 0xFF
                subchunk_blocks[sub][idx] = block_id
    out = bytearray()
    out.append(subchunk_count)
    for blocks in subchunk_blocks:
        out.append(0)
        out.extend(blocks)
        out.extend(b"\x00" * (width * depth))
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
            for child in world_out.iterdir():
                if child.is_file():
                    child.unlink()
                else:
                    shutil.rmtree(child)
        else:
            raise FileExistsError(
                "output directory already exists, pass --delete-out to overwrite",
            )
    world_out.mkdir(parents=True, exist_ok=True)
    with open(Path(__file__).parent / "data" / "blocks.json") as blocks_file:
        raw_blocks = json.load(blocks_file)
    blocks = parse_block_json(raw_blocks)
    inverse_block_map = build_inverse_block_map(blocks)
    cdb_output = world_out / "db" / "cdb"
    builder = CdbBuilder(cdb_output, (7, 0))
    converted_chunks = 0
    missing_xpos_chunks = 0
    max_workers = max(1, (os.cpu_count() or 1) - 1)
    logger.info("Using %d worker process(es) for Java chunk conversion", max_workers)
    if max_workers == 1:
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
    builder.write()
    if converted_chunks == 0 and missing_xpos_chunks > 0:
        logger.error(
            "Conversion failed: all %d chunks were skipped because they were missing an xPos tag",
            missing_xpos_chunks,
        )
    logger.info("Conversion complete: %d chunks converted", converted_chunks)
