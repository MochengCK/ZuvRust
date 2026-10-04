//! **瓦片数据的切分**（规范 §5.9.3 `tile_group_obu`）—— 自研软解 M3 的第二半。
//!
//! 一个 `OBU_FRAME` 的载荷 = 非压缩头 + `byte_alignment()` + **瓦片组**。
//! 瓦片组自己是很短的一段明文头（`tile_start_and_end_present_flag` +
//! `byte_alignment`），后面按 `TileSizeBytes` 一个个切出瓦片的字节 ——
//! 每个瓦片的字节就是它那份**独立的算术编码流**（每个瓦片一份 CDF 状态，
//! 见 `entropy::SymbolDecoder`）。
//!
//! 切分的权威依据是 `tile_info()`（`params.rs`）：
//!
//! ```text
//! tile_group_obu( sz ) {
//!   NumTiles = TileCols * TileRows
//!   if ( NumTiles > 1 ) tile_start_and_end_present_flag f(1)
//!   if ( 单瓦片 或 标志为 0 ) { tg_start = 0; tg_end = NumTiles - 1 }
//!   else { tileBits = TileColsLog2 + TileRowsLog2; tg_start f(tileBits); tg_end f(tileBits) }
//!   byte_alignment()
//!   sz -= headerBytes
//!   for ( TileNum = tg_start; TileNum <= tg_end; TileNum++ ) {
//!     lastTile = (TileNum == tg_end)
//!     if ( lastTile ) tileSize = sz                       // 末瓦片吃光剩下的
//!     else { tile_size_minus_1 le(TileSizeBytes); tileSize = tile_size_minus_1 + 1; sz -= tileSize + TileSizeBytes }
//!     …MiRowStart/MiRowEnd/MiColStart/MiColEnd…
//!   }
//! }
//! ```
//!
//! 注意 `le(TileSizeBytes)` 是**小端**（§4.10.4），与引擎里其它"读字节"的地方
//! 都不同 —— 抄成大端的话，单瓦片流看不出问题（末瓦片不读尺寸字段），
//! 多瓦片流立刻崩。
//!
//! 为什么这段值得单独做：它一边用 `tile_info()` 算出的瓦片网格，一边按
//! "每个瓦片声明的长度"切开整段数据，**所有瓦片加起来必须正好用完整段数据**
//! —— 这就是"非压缩头 + 瓦片组头部读对了"的直接证据（真素材 36845 个瓦片组全过）。

use me_core::{MediaError, Result};

use super::bitreader::BitReader;
use super::params::TileInfo;

/// 一个瓦片：它的字节数据 + 它在帧里的位置（模式信息行列范围）。
///
/// `mi_*` 是"模式信息单元"（4×4 亮度样本）的下标范围，`decode_tile()` 就是按
/// 这个范围对超块做双重循环（规范 §5.11.1）。M4 的 `decode_partition` 直接用它。
#[derive(Debug, Clone)]
pub struct Tile<'a> {
    /// 瓦片序号（`tg_start..=tg_end` 里的 `TileNum`）
    pub index: u32,
    /// 瓦片行（`TileNum / TileCols`）
    pub row: u32,
    /// 瓦片列（`TileNum % TileCols`）
    pub col: u32,
    pub mi_row_start: u32,
    pub mi_row_end: u32,
    pub mi_col_start: u32,
    pub mi_col_end: u32,
    /// 这个瓦片的**独立算术编码流**
    pub data: &'a [u8],
}

/// 切分出来的瓦片组。
#[derive(Debug, Clone)]
pub struct TileGroup<'a> {
    /// 本组覆盖的瓦片序号范围（`tile_start_and_end_present_flag` 为 0 时是全部）
    pub tg_start: u32,
    pub tg_end: u32,
    /// 瓦片组头部占了多少字节（含 `byte_alignment`）
    pub header_bytes: usize,
    pub tiles: Vec<Tile<'a>>,
}

/// 解析 `tile_group_obu( sz )`：读头部、按尺寸字段切出每个瓦片。
///
/// `payload` 传整个 OBU 载荷（`sz` 就是 `payload.len()`），
/// `ti` 是这一帧的 `tile_info()`。
pub fn parse_tile_group<'a>(payload: &'a [u8], ti: &TileInfo) -> Result<TileGroup<'a>> {
    let num_tiles = ti.num_tiles();
    if num_tiles == 0 {
        return Err(MediaError::Bitstream("瓦片数是 0（tile_info 没读对）".into()));
    }

    let mut r = BitReader::new(payload);
    let mut tg_start = 0u32;
    let mut tg_end = num_tiles - 1;
    if num_tiles > 1 {
        // 注意：`tile_start_and_end_present_flag = 0` 在 NumTiles == 1 时**不读位**
        if r.bit()? {
            let bits = ti.cols_log2 + ti.rows_log2;
            tg_start = r.f(bits)?;
            tg_end = r.f(bits)?;
        }
    }
    if tg_start > tg_end || tg_end >= num_tiles {
        return Err(MediaError::Bitstream(format!(
            "tile group 的瓦片范围不合法：tg_start={tg_start}、tg_end={tg_end}（共 {num_tiles} 个瓦片）"
        )));
    }
    // byte_alignment()：填充位必须是 0
    r.byte_align()?;
    let header_bytes = r.bits_read() / 8;
    if header_bytes > payload.len() {
        return Err(MediaError::NeedMore { need: header_bytes, have: payload.len() });
    }
    let body = &payload[header_bytes..];

    let size_bytes = ti.tile_size_bytes as usize;
    let count = (tg_end - tg_start + 1) as usize;
    let mut tiles = Vec::with_capacity(count);
    let mut off = 0usize;
    for i in 0..count {
        let is_last = i + 1 == count;
        let tile_size = if is_last {
            // 末瓦片吃掉剩下的全部（规范：tileSize = sz）
            body.len() - off
        } else {
            if size_bytes == 0 {
                return Err(MediaError::Bitstream(
                    "瓦片组里有多个瓦片，但 tile_info 给的 TileSizeBytes = 0".into(),
                ));
            }
            if off + size_bytes > body.len() {
                return Err(MediaError::NeedMore {
                    need: size_bytes,
                    have: body.len() - off,
                });
            }
            // `le( TileSizeBytes )`：**小端**（§4.10.4）
            let mut size = 0u32;
            for k in 0..size_bytes {
                size |= (body[off + k] as u32) << (8 * k);
            }
            off += size_bytes;
            let size = size as usize + 1;
            if off + size > body.len() {
                return Err(MediaError::Bitstream(format!(
                    "第 {i} 个瓦片声明 {size} 字节，但只剩 {} 字节（tile_size_bytes 或对齐读错了？）",
                    body.len() - off
                )));
            }
            size
        };
        let tile_num = tg_start + i as u32;
        let row = tile_num / ti.cols.max(1);
        let col = tile_num % ti.cols.max(1);
        let (mi_row_start, mi_row_end) = (
            *ti.mi_row_starts.get(row as usize).unwrap_or(&0),
            *ti.mi_row_starts.get(row as usize + 1).unwrap_or(&0),
        );
        let (mi_col_start, mi_col_end) = (
            *ti.mi_col_starts.get(col as usize).unwrap_or(&0),
            *ti.mi_col_starts.get(col as usize + 1).unwrap_or(&0),
        );
        tiles.push(Tile {
            index: tile_num,
            row,
            col,
            mi_row_start,
            mi_row_end,
            mi_col_start,
            mi_col_end,
            data: &body[off..off + tile_size],
        });
        off += tile_size;
    }
    if off != body.len() {
        return Err(MediaError::Bitstream(format!(
            "瓦片数据没用完：切到 {off}、总长 {}",
            body.len()
        )));
    }
    Ok(TileGroup { tg_start, tg_end, header_bytes, tiles })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 单瓦片：`tile_info` 只有一个瓦片时，瓦片组头部**不读任何位**（直接对齐）。
    fn single_tile_info() -> TileInfo {
        TileInfo {
            uniform: true,
            cols: 1,
            rows: 1,
            cols_log2: 0,
            rows_log2: 0,
            context_update_tile_id: 0,
            tile_size_bytes: 0,
            mi_col_starts: vec![0, 480],
            mi_row_starts: vec![0, 270],
            sb_cols: 30,
            sb_rows: 17,
            sb_shift: 4,
        }
    }

    #[test]
    fn single_tile_takes_the_whole_payload() {
        let ti = single_tile_info();
        let payload: Vec<u8> = (0..64u32).map(|i| i as u8).collect();
        let g = parse_tile_group(&payload, &ti).unwrap();
        assert_eq!(g.header_bytes, 0, "单瓦片不需要头部位");
        assert_eq!(g.tiles.len(), 1);
        assert_eq!(g.tiles[0].data, &payload[..]);
        assert_eq!(g.tiles[0].mi_row_start, 0);
        assert_eq!(g.tiles[0].mi_row_end, 270);
        assert_eq!(g.tiles[0].mi_col_end, 480);
    }

    #[test]
    fn two_tiles_read_little_endian_sizes() {
        // 2 列 × 1 行，尺寸字段 2 字节
        let ti = TileInfo {
            uniform: false,
            cols: 2,
            rows: 1,
            cols_log2: 1,
            rows_log2: 0,
            tile_size_bytes: 2,
            mi_col_starts: vec![0, 240, 480],
            mi_row_starts: vec![0, 270],
            ..single_tile_info()
        };
        // 瓦片组头部：NumTiles = 2 → 读 1 位 flag = 0（表示覆盖全部）→ 再对齐
        // 体：[size(2B, LE)] [tile0] [tile1]
        let tile0 = [0xAAu8, 0xBB, 0xCC]; // 3 字节 → size_minus_1 = 2
        let tile1 = [0x11u8, 0x22]; // 剩下的全归末瓦片
        let mut payload = vec![0u8]; // 头部 1 字节（flag=0 + 7 位补 0）
        payload.extend_from_slice(&2u16.to_le_bytes()); // le(2) = 2
        payload.extend_from_slice(&tile0);
        payload.extend_from_slice(&tile1);
        let g = parse_tile_group(&payload, &ti).unwrap();
        assert_eq!(g.header_bytes, 1);
        assert_eq!(g.tiles.len(), 2);
        assert_eq!(g.tiles[0].data, &tile0[..]);
        assert_eq!(g.tiles[1].data, &tile1[..]);
        assert_eq!(g.tiles[0].col, 0);
        assert_eq!(g.tiles[1].col, 1);
        assert_eq!(g.tiles[1].mi_col_start, 240);
    }

    #[test]
    fn sizes_are_little_endian_not_big_endian() {
        // 尺寸字段 2 字节时，值 2 的字节序是 `[0x02, 0x00]`（小端，§4.10.4）。
        // 若按大端读，`[0x02, 0x00]` 会得到 0x0200 = 512 → 立刻越界报错。
        // 这条测试就是"le 不是 be"的守卫（多瓦片流才会走到这段）。
        let ti = TileInfo {
            cols: 2,
            rows: 1,
            cols_log2: 1,
            tile_size_bytes: 2,
            mi_col_starts: vec![0, 240, 480],
            mi_row_starts: vec![0, 270],
            ..single_tile_info()
        };
        let mut payload = vec![0u8];
        payload.extend_from_slice(&[0x02, 0x00]); // 小端 = 2（正确写法）
        payload.extend_from_slice(&[0u8; 8]);
        let g = parse_tile_group(&payload, &ti).unwrap();
        assert_eq!(g.tiles[0].data.len(), 3, "le(2) + 1 = 3 字节");
        assert_eq!(g.tiles[1].data.len(), 5, "末瓦片吃掉剩下的（10 - 2 - 3）");

        // 换成大端写法 `[0x00, 0x02]`（小端读 = 512 → tile_size = 513）必须报越界
        let mut bad = vec![0u8];
        bad.extend_from_slice(&[0x00, 0x02]);
        bad.extend_from_slice(&[0u8; 8]);
        let e = parse_tile_group(&bad, &ti).unwrap_err();
        assert!(
            format!("{e}").contains("513"),
            "按小端读 0x0200 = 512，+1 后应当报 513 越界：{e}"
        );
    }

    #[test]
    fn tile_range_out_of_bounds_is_rejected() {
        let ti = TileInfo {
            cols: 2,
            rows: 2,
            cols_log2: 1,
            rows_log2: 1,
            tile_size_bytes: 1,
            mi_col_starts: vec![0, 240, 480],
            mi_row_starts: vec![0, 135, 270],
            ..single_tile_info()
        };
        // NumTiles = 4 → 读 flag = 1 → tileBits = 2 → tg_start = 3、tg_end = 1（非法）
        let mut w = crate::video::testbits::BitWriter::default();
        w.f(1, 1); // tile_start_and_end_present_flag
        w.f(3, 2); // tg_start
        w.f(1, 2); // tg_end
        w.byte_align();
        let payload = w.finish().to_vec();
        let e = parse_tile_group(&payload, &ti).unwrap_err();
        assert!(format!("{e}").contains("范围不合法"), "{e}");
    }
}
