// #pragma once
// #include <stdint.h>

/*
    all-in-one header for Minecraft 3DS world files

    - shared stuff (FileHeader etc) at the top
    - CDB = chunk DB (blocks, subchunks, etc.)
    - VDB = global DB (player, maps, biomes, structures...)

    goal: match the game’s expectations exactly so it doesn’t wipe
          your converted files on load
*/

#define MAGIC_CDB 0xABCDEF98
#define MAGIC_VDB 0xABCDEF99

// #pragma pack(push, 1)

/* --------------------------------------------------
   SHARED CONTAINER HEADER (sltXX.cdb / sltXX.vdb)

   NOTE:
   - this is used only for the slot files: sltNN.cdb, sltNN.vdb
   - index.cdb / newindex.cdb and index.vdb / newindex.vdb
     DO NOT have this, they start directly with their index header
   -------------------------------------------------- */

struct FileHeader {
    uint16_t something0;   // always 1
    uint16_t something1;   // always 1

    uint32_t subfileCount; // how many subfiles/pages
    uint32_t headerSize;   // bytes before first subfile; 0x14 in samples
    uint32_t subfileSize;  // page size; CDB ~0x2800, VDB 0x1000/0x2000/0x20000
    uint32_t fileType;     // 0x00000004 = CDB, 0x00000100 = VDB
};

/* start of each subfile/page inside slt*.cdb / slt*.vdb */
struct SubfileHeader {
    uint32_t magic;        // MAGIC_CDB or MAGIC_VDB (for valid pages)
};



/* ==================================================
   CDB SIDE (chunk DB, blocks, subchunks, etc.)
   ================================================== */

/* --------------------------------------
   chunk position packing (index + chunk)
   -------------------------------------- */

struct Position {
    uint32_t x : 14;        // signed 14-bit
    uint32_t z : 14;        // signed 14-bit
    uint32_t dimension : 4; // 0 = overworld, 1 = nether, etc.
};

/* tiny helper, optional in your code */
// static inline uint32_t pos_pack(int x, int z, int dim) {
//     return ((uint32_t)(x & 0x3FFF))
//          | ((uint32_t)(z & 0x3FFF) << 14)
//          | ((uint32_t)(dim & 0xF)  << 28);
// }


/* --------------------------------------
   Chunk header inside each CDB subfile
   -------------------------------------- */

struct ChunkParameters {
    int8_t unknown0;   // usually 1..7; seems like "chunk status"
    int8_t unknown1;   // always 0 in worlds we saw
};

struct ChunkSection {
    int32_t index;            // -1 = unused section; otherwise 0..n
    int32_t position;         // offset from subfile start (absolute), or -1
    int32_t compressedSize;   // bytes of compressed blob; 0 = empty
    int32_t decompressedSize; // size after zlib; 0 = empty
};

struct ChunkHeader {
    struct Position position;       // must match CDBEntry.position exactly
    struct ChunkParameters params;  // same as in CDBEntry.parameters

    uint16_t unknown0;              // 0 in samples
    uint16_t unknown1;              // always 3 so far
    uint16_t unknown2;              // sometimes 0 / sometimes not (heightmap-ish)

    struct ChunkSection sections[6]; // up to 6 blobs (subchunks + extra data)
};

/*
   CDB slot file layout (sltNN.cdb):

   FileHeader fh;   // at offset 0
   then for i in [0..fh.subfileCount-1]:
       at offset fh.headerSize + i * fh.subfileSize:

       SubfileHeader sh;
         sh.magic MUST be MAGIC_CDB if that page used, or 0 for totally empty

       ChunkHeader ch;
         ch.position must match the CDBEntry for that (slot,subfile)
         ch.params must match CDBEntry.parameters

       then compressed data for each sections[k] with index != -1 and compressedSize > 0
       then padding up to fh.subfileSize
*/


/* --------------------------------------
   CDB index: index.cdb / newindex.cdb
   IMPORTANT:
   - NO FileHeader here
   - file starts directly with CdbIndexHeader
   -------------------------------------- */

struct CdbIndexHeader {
    uint32_t constant0;     // always 0x00000002
    uint32_t entryCount;    // number of CDBEntry at the end
    uint32_t unknown0;      // 0x00003E04 in sample; treat as opaque
    uint32_t entrySize;     // MUST be 16 (sizeof CDBEntry with pack(1))
    uint32_t pointerCount;  // 28 in sample worlds
    uint32_t constant1;     // always 0x00000080
};

/* pointer table right after CdbIndexHeader, size = pointerCount * 4
   in stock files they are 0..27, so basically just identity
*/
struct IndexPointer {
    uint32_t value;         // slot-group id; for CDB strictly 0..pointerCount-1
};

/*
   CDB index entry: 16 bytes, repeated entryCount times
   directly after the pointer array.

   This is what connects a chunk to a slot/subfile.
   The constants are very strict; if you get them wrong,
   the game will likely re-create all CDBs.
*/

struct CDBEntry {
    struct Position position; // packed x/z/dim, must match chunk header

    uint16_t slot;            // which sltXX.cdb (use this number for XX)
    uint16_t subfile;         // page index inside that slot

    uint16_t constant0;       // always 0x20FF
    uint16_t constant1;       // always 0x000A

    struct ChunkParameters parameters; // same two bytes as ChunkHeader.params

    uint16_t constant2;       // always 0x8000
};

/*
   Checklist if you’re generating index.cdb:

   - DO NOT put FileHeader in front of it
   - sizeof(CdbIndexHeader) + pointerCount*4 + entryCount*entrySize == file_size
   - CdbIndexHeader.constant0 == 2
   - CdbIndexHeader.constant1 == 0x80
   - CdbIndexHeader.entrySize == 16
   - pointer[i].value == i (or at least coherent with how you lay out slots)
   - For each CDBEntry:
       slot -> slt<slot>.cdb must exist
       subfile < FileHeader.subfileCount in that slt
       ChunkHeader.position == CDBEntry.position
       ChunkHeader.params   == CDBEntry.parameters
       constant0 == 0x20FF, constant1 == 0x000A, constant2 == 0x8000
*/


/* --------------------------------------
   Decompressed block data shape (approx)
   useful when inflating sections[] blobs
   -------------------------------------- */

struct Subchunk {
    uint8_t constant0;                       // always 0x0 in samples
    uint8_t blocks[16][16][16];              // block ids
    uint8_t blockData[16 * 16 * 16 / 2];     // nibbles, usually meta
    uint8_t unknownBlockData[16][16][16];    // seems all 0 so far
};

struct BlockData {
    uint8_t subchunkCount;
    struct Subchunk subchunks[];             // length = subchunkCount
    /* then:
       uint16_t unknown0[16][16];            // maybe heightmap/data
       uint8_t  biomes[16][16];              // biome ids
    */
};



/* ==================================================
   VDB SIDE (global DB: player, maps, biomes, etc.)
   ================================================== */

/*
   VDB slot files: sltNN.vdb

   - have the same FileHeader as CDB, but fileType = 0x100
   - subfiles/pages also start with MAGIC_VDB
   - inside each page there is *one* VDB record, aligned to page start
   - record must not cross page boundary
*/


/* --------------------------------------
   VDB record layout (inside a page)
   -------------------------------------- */

enum VdbKind {
    VDB_KIND_UNKNOWN          = 0,
    VDB_KIND_MAP              = 2, // map_<id> records
    VDB_KIND_GLOBAL_GENERIC   = 3, // ~local_player, Overworld, dimension0, portals, BiomeData copy
    VDB_KIND_BIOME_CANONICAL  = 4, // canonical BiomeData in its own slot
    VDB_KIND_STRUCTURES       = 6  // AutonomousEntities, mVillages, etc.
};

/*
   One record looks like:

   uint32_t magic;       // MAGIC_VDB (0xABCDEF99)
   uint32_t keyLen;      // length of the key (no NUL)
   uint32_t flags;       // 0 in all seen files (reserved)
   uint32_t kind;        // see VdbKind above

   char     key[keyLen]; // ASCII, e.g. "~local_player", "map_-38654705599"
   uint32_t payloadLen;  // NBT length
   uint8_t  payload[payloadLen]; // little-endian NBT, Bedrock-style

   total size of record must fit inside that subfile/page.
*/

struct VdbRecordHeader {
    uint32_t magic;   // MAGIC_VDB
    uint32_t keyLen;  // bytes, no terminator
    uint32_t flags;   // reserved, 0 so far
    uint32_t kind;    // VdbKind or similar
    /* then:
       char    key[keyLen];
       uint32_t payloadLen;
       uint8_t  payload[payloadLen];
    */
};

/*
   When editing / regenerating:

   - pageOffset = fh.headerSize + pageIndex * fh.subfileSize
   - place VdbRecordHeader at pageOffset
   - write key, payloadLen, payload right after
   - recordSize = 4+4+4+4 + keyLen + 4 + payloadLen
   - recordSize <= fh.subfileSize
   - no need to touch any separate offset table; index.vdb only cares
     about which slot/group, not the exact byte offset inside page
*/


/* --------------------------------------
   VDB index files: index.vdb / newindex.vdb

   again: NO FileHeader here, file starts right
   with the VdbIndexHeader
   -------------------------------------- */

struct VdbIndexHeader {
    uint32_t constant0;     // always 0x2
    uint32_t entryCount;    // number of VdbIndexEntry
    uint32_t unknown0;      // 0x80 in samples
    uint32_t entrySize;     // 0x10C (268 bytes)
    uint32_t pointerCount;  // 10 (index.vdb) or 15 (newindex.vdb)
    uint32_t constant1;     // always 0x80

    /* then:
       IndexPointer pointers[pointerCount]; // same struct as CDB index, reused
       VdbIndexEntry entries[entryCount];
    */
};

/*
   For VDB, pointers[] values look like "slot-groups" that map to sltNN.vdb.
   The entries array then uses some of the fields[] to refer to those groups
   and pick which page.
*/

struct VdbIndexEntry {
    char     name[48];   // zero-terminated key name, e.g. "~local_player"
    uint32_t fields[55]; // raw metadata, 55 * 4 = 220 bytes
};

/*
   Only partially understood, but enough for not breaking it:

   - name must match the key string used in VDB (exact same UTF-8)
   - fields[x] contain:
       - kind/category bits in the last field (0x80000006, 0x80000003, etc.)
       - slot-group id that points into pointer[] / sltNN.vdb family
       - some sort of page index / offset
       - also runtime pointers when the world was saved

   For simply *preserving* a world, you don't need to rewrite this.
   For tool-generated worlds, easiest: copy an existing entry for same type
   (player/map/biome) and only tweak what’s absolutely necessary
   (e.g. name, maybe id inside name for map_<id>).

   Most of the heavy lifting for real correctness is in:
   - the VDB record in the slt file itself (key + NBT)
   - the CDB side for blocks
*/


/* --------------------------------------
   tiny summary of invariants to keep
   so the game doesn't eat your files
   -------------------------------------- */

/*
   CDB:
   - index.cdb:
       no FileHeader at start
       header.constant0  == 2
       header.constant1  == 0x80
       header.entrySize  == 16
       file size matches header/pointerCount/entryCount
       pointers[i].value == i (simple case)
   - each CDBEntry:
       constant0 == 0x20FF
       constant1 == 0x000A
       constant2 == 0x8000
       slot/subfile valid and pointing to a real MAGIC_CDB subfile
       Position + parameters match ChunkHeader at that subfile

   VDB:
   - slt*.vdb:
       FileHeader.fileType == 0x100
       each used page starts with magic == MAGIC_VDB
       exactly one record per page, aligned at page start
       record size never crosses page boundary
   - index.vdb/newindex.vdb:
       same pattern as CDB index: constant0 == 2, constant1 == 0x80,
       entrySize == 0x10C, no FileHeader at start.
       names match keys exactly.
*/

// #pragma pack(pop)
