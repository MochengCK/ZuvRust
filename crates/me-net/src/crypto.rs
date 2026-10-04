//! 自研对称加密原语：AES（128/192/256，ECB/CBC/CTR）+ Base64 + PKCS7。
//!
//! 为什么自己写：本工程不许引入第三方 crypto crate。AES 是**唯一**需要自己实现的
//! 密码学算法，而且它的输入域非常窄（HLS 的 `EXT-X-KEY` 只用 AES-128-CBC 与
//! AES-128-CTR，个别源用 AES-256），实现正确性完全可以用官方向量钉死。
//!
//! 几条刻意的设计决定：
//!
//! 1. **状态布局按 FIPS-197 的列主序**：`state[r + 4*c] = in[4*c + r]`。
//!    很多人写成行主序也能自洽（加密解密配对就行），但那样**官方测试向量一定对不上**，
//!    所以这里严格照抄规范，并用 FIPS-197 / SP 800-38A 的向量做回归。
//! 2. **`aes_cbc_decrypt` 的 padding 非法时报错，不静默截断**。CBC 下 padding 坏了
//!    说明密钥不对或密文被篡改 —— 静默截断会让播放器拿到一段"看起来合法"的乱码，
//!    排查成本极高。宁可立刻报错。
//! 3. **密钥展开只做一次**（`expand` 在 cbc/ctr/ecb 入口调一次），不是每块展开一次：
//!    一个 2MB 的分片有 13 万个块，每块重算 11 轮子密钥是纯浪费。
//! 4. **`decode_key_body` 要容忍三种真实形态**：裸 16 字节、32 字符 hex、24 字符 Base64。
//!    同一份 HLS 生态里三种都有，而且很多站点的密钥 URI 会带尾随换行。
//! 5. **百分号/URL-safe Base64 一起认**：HLS 密钥经常是 URL-safe Base64（`-` `_`），
//!    如果只认标准字母表就会把合法密钥判成非法。

use me_core::error::{MediaError, Result};

// ---------------------------------------------------------------------------
// AES 查表与基础运算
// ---------------------------------------------------------------------------

/// AES S-box（FIPS-197 Figure 7）。
const SBOX: [u8; 256] = [
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
];

/// 逆 S-box（FIPS-197 Figure 14）。解密用。
const INV_SBOX: [u8; 256] = [
    0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
    0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
    0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
    0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
    0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
    0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
    0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
    0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
    0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
    0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
    0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
    0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
    0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
    0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
    0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d,
];

/// 轮常量 RCON（FIPS-197 §5.2）。索引 1..=10 有效，0 位空置让下标即轮号。
const RCON: [u8; 11] = [
    0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36,
];

/// GF(2^8) 上的 `xtime`（乘 2）。
#[inline]
fn xtime(b: u8) -> u8 {
    // 最高位为 1 时左移后要异或不可约多项式 x^8+x^4+x^3+x+1 的低 8 位 0x1b。
    (b << 1) ^ (((b >> 7) & 1) * 0x1b)
}

/// GF(2^8) 上的乘法（用于 MixColumns）。AES 的域很小，朴素移位加法足够快，
/// 也省掉了 256×256 查表带来的缓存压力。
#[inline]
fn gf_mul(a: u8, b: u8) -> u8 {
    let mut a = a;
    let mut b = b;
    let mut r = 0u8;
    while b != 0 {
        if b & 1 != 0 {
            r ^= a;
        }
        a = xtime(a);
        b >>= 1;
    }
    r
}

// ---------------------------------------------------------------------------
// 密钥
// ---------------------------------------------------------------------------

/// AES 密钥。三档长度对应 10/12/14 轮。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AesKey {
    K128([u8; 16]),
    K192([u8; 24]),
    K256([u8; 32]),
}

impl AesKey {
    /// 从裸字节构造。只有 16/24/32 三种长度合法，其它长度一律报错
    /// （不能"补齐到 16 字节" —— 补齐后的密钥解出来是乱码，不如早报错）。
    pub fn from_bytes(b: &[u8]) -> Result<AesKey> {
        match b.len() {
            16 => {
                let mut k = [0u8; 16];
                k.copy_from_slice(b);
                Ok(AesKey::K128(k))
            }
            24 => {
                let mut k = [0u8; 24];
                k.copy_from_slice(b);
                Ok(AesKey::K192(k))
            }
            32 => {
                let mut k = [0u8; 32];
                k.copy_from_slice(b);
                Ok(AesKey::K256(k))
            }
            n => Err(MediaError::protocol(format!(
                "AES 密钥长度非法: {n} 字节（只接受 16/24/32）"
            ))),
        }
    }

    /// 裸密钥字节。
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            AesKey::K128(k) => k,
            AesKey::K192(k) => k,
            AesKey::K256(k) => k,
        }
    }

    /// 加密单个 16 字节块（就地）。
    pub fn encrypt_block(&self, block: &mut [u8; 16]) {
        let rk = self.expand();
        encrypt_block_with(&rk, block);
    }

    /// 解密单个 16 字节块（就地）。
    pub fn decrypt_block(&self, block: &mut [u8; 16]) {
        let rk = self.expand();
        decrypt_block_with(&rk, block);
    }

    /// 密钥展开：把 Nk 个 32bit 字扩成 4*(Nr+1) 个字，按轮切成 [u8;16]。
    ///
    /// 展开一次给整个分片复用；`encrypt_block` 这种单块入口才每次现算。
    pub fn expand(&self) -> Vec<[u8; 16]> {
        let key = self.as_bytes();
        let nk = key.len() / 4;
        let nr = nk + 6;
        let total_words = 4 * (nr + 1);

        let mut w = vec![0u32; total_words];
        for i in 0..nk {
            w[i] = u32::from_be_bytes([key[4 * i], key[4 * i + 1], key[4 * i + 2], key[4 * i + 3]]);
        }
        for i in nk..total_words {
            let mut temp = w[i - 1];
            if i % nk == 0 {
                // RotWord + SubWord + Rcon
                temp = sub_word(temp.rotate_left(8)) ^ ((RCON[i / nk] as u32) << 24);
            } else if nk > 6 && i % nk == 4 {
                // AES-256 独有：中间再做一次 SubWord（FIPS-197 §5.2 的特殊情形）
                temp = sub_word(temp);
            }
            w[i] = w[i - nk] ^ temp;
        }

        let mut rounds = Vec::with_capacity(nr + 1);
        for r in 0..=nr {
            let mut rk = [0u8; 16];
            for c in 0..4 {
                rk[4 * c..4 * c + 4].copy_from_slice(&w[4 * r + c].to_be_bytes());
            }
            rounds.push(rk);
        }
        rounds
    }
}

fn sub_word(w: u32) -> u32 {
    let b = w.to_be_bytes();
    u32::from_be_bytes([
        SBOX[b[0] as usize],
        SBOX[b[1] as usize],
        SBOX[b[2] as usize],
        SBOX[b[3] as usize],
    ])
}

// ---------------------------------------------------------------------------
// 块级加解密（列主序，严格照 FIPS-197）
// ---------------------------------------------------------------------------

#[inline]
fn add_round_key(state: &mut [u8; 16], rk: &[u8; 16]) {
    for i in 0..16 {
        state[i] ^= rk[i];
    }
}

#[inline]
fn sub_bytes(state: &mut [u8; 16]) {
    for b in state.iter_mut() {
        *b = SBOX[*b as usize];
    }
}

#[inline]
fn inv_sub_bytes(state: &mut [u8; 16]) {
    for b in state.iter_mut() {
        *b = INV_SBOX[*b as usize];
    }
}

/// 行移位：第 r 行循环左移 r 个字节。`state[r + 4c]` 是列主序下标。
#[inline]
fn shift_rows(state: &mut [u8; 16]) {
    let s = *state;
    for r in 1..4 {
        for c in 0..4 {
            state[r + 4 * c] = s[r + 4 * ((c + r) % 4)];
        }
    }
}

#[inline]
fn inv_shift_rows(state: &mut [u8; 16]) {
    let s = *state;
    for r in 1..4 {
        for c in 0..4 {
            // 反向：右移 r 等价于取左边第 (c - r) 个
            state[r + 4 * c] = s[r + 4 * ((c + 4 - r) % 4)];
        }
    }
}

#[inline]
fn mix_columns(state: &mut [u8; 16]) {
    for c in 0..4 {
        let i = 4 * c;
        let (a0, a1, a2, a3) = (state[i], state[i + 1], state[i + 2], state[i + 3]);
        state[i] = xtime(a0) ^ (xtime(a1) ^ a1) ^ a2 ^ a3;
        state[i + 1] = a0 ^ xtime(a1) ^ (xtime(a2) ^ a2) ^ a3;
        state[i + 2] = a0 ^ a1 ^ xtime(a2) ^ (xtime(a3) ^ a3);
        state[i + 3] = (xtime(a0) ^ a0) ^ a1 ^ a2 ^ xtime(a3);
    }
}

#[inline]
fn inv_mix_columns(state: &mut [u8; 16]) {
    for c in 0..4 {
        let i = 4 * c;
        let (a0, a1, a2, a3) = (state[i], state[i + 1], state[i + 2], state[i + 3]);
        state[i] = gf_mul(a0, 14) ^ gf_mul(a1, 11) ^ gf_mul(a2, 13) ^ gf_mul(a3, 9);
        state[i + 1] = gf_mul(a0, 9) ^ gf_mul(a1, 14) ^ gf_mul(a2, 11) ^ gf_mul(a3, 13);
        state[i + 2] = gf_mul(a0, 13) ^ gf_mul(a1, 9) ^ gf_mul(a2, 14) ^ gf_mul(a3, 11);
        state[i + 3] = gf_mul(a0, 11) ^ gf_mul(a1, 13) ^ gf_mul(a2, 9) ^ gf_mul(a3, 14);
    }
}

fn encrypt_block_with(rk: &[[u8; 16]], block: &mut [u8; 16]) {
    let nr = rk.len() - 1;
    add_round_key(block, &rk[0]);
    for round in 1..nr {
        sub_bytes(block);
        shift_rows(block);
        mix_columns(block);
        add_round_key(block, &rk[round]);
    }
    // 最后一轮不做 MixColumns
    sub_bytes(block);
    shift_rows(block);
    add_round_key(block, &rk[nr]);
}

fn decrypt_block_with(rk: &[[u8; 16]], block: &mut [u8; 16]) {
    let nr = rk.len() - 1;
    add_round_key(block, &rk[nr]);
    for round in (1..nr).rev() {
        inv_shift_rows(block);
        inv_sub_bytes(block);
        add_round_key(block, &rk[round]);
        inv_mix_columns(block);
    }
    inv_shift_rows(block);
    inv_sub_bytes(block);
    add_round_key(block, &rk[0]);
}

// ---------------------------------------------------------------------------
// 分组模式
// ---------------------------------------------------------------------------

/// 裸 CBC 解密（**不做 PKCS7 去填充**）。
///
/// 之所以把它单独拎出来：官方 NIST 向量的明文长度恰好是整块且末字节碰巧是 0x10，
/// 走 `pkcs7_unpad` 会被误判成填充。测试要用它去对官方向量，
/// 而生产路径一律走 [`aes_cbc_decrypt`]（带去填充与校验）。
pub(crate) fn aes_cbc_decrypt_raw(key: &AesKey, iv: &[u8; 16], data: &[u8]) -> Result<Vec<u8>> {
    if data.is_empty() {
        return Ok(Vec::new());
    }
    if data.len() % 16 != 0 {
        return Err(MediaError::protocol(format!(
            "CBC 密文长度必须是 16 的倍数，实际 {}",
            data.len()
        )));
    }
    let rk = key.expand();
    let mut out = vec![0u8; data.len()];
    let mut prev = *iv;
    for (i, chunk) in data.chunks_exact(16).enumerate() {
        let mut block = [0u8; 16];
        block.copy_from_slice(chunk);
        let cipher_block = block;
        decrypt_block_with(&rk, &mut block);
        for j in 0..16 {
            out[i * 16 + j] = block[j] ^ prev[j];
        }
        prev = cipher_block;
    }
    Ok(out)
}

/// AES-CBC 解密 + PKCS7 去填充（HLS `METHOD=AES-128` 的主路径）。
///
/// 填充非法直接报错：这是密钥错误/密文被篡改的信号，静默截断只会让问题后移。
pub fn aes_cbc_decrypt(key: &AesKey, iv: &[u8; 16], data: &[u8]) -> Result<Vec<u8>> {
    if data.is_empty() {
        return Err(MediaError::protocol("CBC 密文为空"));
    }
    let plain = aes_cbc_decrypt_raw(key, iv, data)?;
    match pkcs7_unpad(&plain) {
        Some(v) => Ok(v.to_vec()),
        None => Err(MediaError::protocol(
            "CBC 去填充失败：密钥不对或密文被篡改",
        )),
    }
}

/// AES-CTR 解密（`METHOD=AES-128` 的另一种模式；加解密同构）。
///
/// 计数器按 NIST SP 800-38A 的约定当作 **128 位大端整数** 递增：
/// 只递增最后几个字节是常见错误，会在跨 2^32 块时错位（长直播流真的会撞上）。
pub fn aes_ctr_decrypt(key: &AesKey, iv: &[u8; 16], data: &[u8]) -> Result<Vec<u8>> {
    let rk = key.expand();
    let mut out = Vec::with_capacity(data.len());
    let mut counter = *iv;
    for chunk in data.chunks(16) {
        let mut ks = counter;
        encrypt_block_with(&rk, &mut ks);
        for (i, b) in chunk.iter().enumerate() {
            out.push(b ^ ks[i]);
        }
        inc_be128(&mut counter);
    }
    Ok(out)
}

/// 128 位大端 +1。
fn inc_be128(c: &mut [u8; 16]) {
    for i in (0..16).rev() {
        let (v, overflow) = c[i].overflowing_add(1);
        c[i] = v;
        if !overflow {
            break;
        }
    }
}

/// AES-ECB 解密：**逐块解密，不做任何填充处理**。
///
/// 保留 ECB 是为了个别私有化 HLS 方言（`METHOD=AES-256` 的变体）与测试对向量；
/// 是否去填充由调用方按自己的方言决定，所以这里不擅自截断。
pub fn aes_ecb_decrypt(key: &AesKey, data: &[u8]) -> Result<Vec<u8>> {
    if data.len() % 16 != 0 {
        return Err(MediaError::protocol(format!(
            "ECB 密文长度必须是 16 的倍数，实际 {}",
            data.len()
        )));
    }
    let rk = key.expand();
    let mut out = vec![0u8; data.len()];
    for (i, chunk) in data.chunks_exact(16).enumerate() {
        let mut block = [0u8; 16];
        block.copy_from_slice(chunk);
        decrypt_block_with(&rk, &mut block);
        out[i * 16..i * 16 + 16].copy_from_slice(&block);
    }
    Ok(out)
}

/// PKCS7 去填充。非法（pad=0 / pad>16 / 填充字节不一致）返回 `None`。
pub fn pkcs7_unpad(data: &[u8]) -> Option<&[u8]> {
    let &pad = data.last()?;
    let pad = pad as usize;
    if pad == 0 || pad > 16 || pad > data.len() {
        return None;
    }
    let split = data.len() - pad;
    if data[split..].iter().all(|&b| b as usize == pad) {
        Some(&data[..split])
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Base64
// ---------------------------------------------------------------------------

const B64_STD: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Base64 解码。宽容但不出错：
/// - 同时接受标准（`+` `/`）与 URL-safe（`-` `_`）字母表；
/// - 忽略空白（很多站点返回的密钥带换行）；
/// - **允许缺省填充**（HLS 密钥常见 22/43 字符的无填充形式）；
/// - 非法字符报错（不像 URL 那样宽容 —— 密钥里出现垃圾字符就该立刻暴露）。
pub fn base64_decode(s: &str) -> Result<Vec<u8>> {
    let mut acc: u32 = 0;
    let mut nbits: u32 = 0;
    let mut out = Vec::with_capacity(s.len() / 4 * 3 + 3);
    let mut seen_pad = false;
    for c in s.chars() {
        if c.is_ascii_whitespace() {
            continue;
        }
        if c == '=' {
            seen_pad = true;
            continue;
        }
        if seen_pad {
            return Err(MediaError::protocol("Base64 填充符后还有数据"));
        }
        let v = match c {
            'A'..='Z' => c as u32 - 'A' as u32,
            'a'..='z' => c as u32 - 'a' as u32 + 26,
            '0'..='9' => c as u32 - '0' as u32 + 52,
            '+' | '-' => 62,
            '/' | '_' => 63,
            other => {
                return Err(MediaError::protocol(format!("Base64 非法字符: {other:?}")));
            }
        };
        acc = (acc << 6) | v;
        nbits += 6;
        if nbits >= 8 {
            nbits -= 8;
            out.push((acc >> nbits) as u8);
        }
    }
    Ok(out)
}

/// Base64 编码（标准字母表 + `=` 填充）。
pub fn base64_encode(b: &[u8]) -> String {
    let mut out = String::with_capacity((b.len() + 2) / 3 * 4);
    for chunk in b.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = *chunk.get(1).unwrap_or(&0) as u32;
        let b2 = *chunk.get(2).unwrap_or(&0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(B64_STD[((n >> 18) & 63) as usize] as char);
        out.push(B64_STD[((n >> 12) & 63) as usize] as char);
        if chunk.len() > 1 {
            out.push(B64_STD[((n >> 6) & 63) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(B64_STD[(n & 63) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}

// ---------------------------------------------------------------------------
// 密钥体归一化
// ---------------------------------------------------------------------------

fn hex_val(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

fn hex_decode(s: &[u8]) -> Option<Vec<u8>> {
    if s.len() % 2 != 0 {
        return None;
    }
    let mut out = Vec::with_capacity(s.len() / 2);
    for pair in s.chunks_exact(2) {
        out.push((hex_val(pair[0])? << 4) | hex_val(pair[1])?);
    }
    Some(out)
}

/// 把 HLS `EXT-X-KEY` URI 取回来的「密钥体」归一成 `want_len` 字节。
///
/// 真实世界里同一个字段有三种写法，且**优先级不能乱**：
/// 1. **裸字节**：长度恰好等于 `want_len` 就当作裸密钥。这条必须最先判，
///    因为 16 字节的裸密钥和 16 字符的 hex 文本长度上有交集，
///    而 HLS 规范里"裸 16 字节"才是标准形态。
/// 2. **hex 文本**：长度 `2*want_len` 且全是 hex 字符（有些源把密钥写成 32 个 hex 字符）。
/// 3. **Base64 文本**：长度约 `ceil(want_len/3)*4`（16 字节 → 24 字符）。
///
/// 顺带修掉尾随 `\n`/空格 —— 服务器用文本文件存密钥时几乎一定带换行，
/// 不 trim 就会得到长度 17 的"密钥"然后报错。
pub fn decode_key_body(body: &[u8], want_len: usize) -> Result<Vec<u8>> {
    let trimmed: &[u8] = {
        let mut start = 0;
        let mut end = body.len();
        while start < end && body[start].is_ascii_whitespace() {
            start += 1;
        }
        while end > start && body[end - 1].is_ascii_whitespace() {
            end -= 1;
        }
        &body[start..end]
    };

    if trimmed.len() == want_len {
        return Ok(trimmed.to_vec());
    }
    if trimmed.len() == want_len * 2 {
        if let Some(v) = hex_decode(trimmed) {
            return Ok(v);
        }
    }
    if let Ok(s) = std::str::from_utf8(trimmed) {
        if let Ok(v) = base64_decode(s) {
            if v.len() == want_len {
                return Ok(v);
            }
        }
    }
    Err(MediaError::protocol(format!(
        "无法识别的密钥体：{} 字节（期望 {} 字节裸密钥 / {} 字符 hex / Base64 文本）",
        trimmed.len(),
        want_len,
        want_len * 2
    )))
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(s: &str) -> Vec<u8> {
        hex_decode(s.as_bytes()).expect("测试向量必须是合法 hex")
    }

    fn arr16(s: &str) -> [u8; 16] {
        let v = hex(s);
        let mut a = [0u8; 16];
        a.copy_from_slice(&v);
        a
    }

    /// FIPS-197 Appendix C 的单块向量：加密方向。
    fn fips197_block(key: &str, ct: &str) {
        let key = AesKey::from_bytes(&hex(key)).unwrap();
        let mut block = arr16("00112233445566778899aabbccddeeff");
        key.encrypt_block(&mut block);
        assert_eq!(
            block.to_vec(),
            hex(ct),
            "FIPS-197 加密向量不匹配（密钥 {} 字节）",
            key.as_bytes().len()
        );
        // 解密方向回得去
        key.decrypt_block(&mut block);
        assert_eq!(block.to_vec(), hex("00112233445566778899aabbccddeeff"));
    }

    #[test]
    fn fips197_c1_aes128() {
        fips197_block(
            "000102030405060708090a0b0c0d0e0f",
            "69c4e0d86a7b0430d8cdb78070b4c55a",
        );
    }

    #[test]
    fn fips197_c2_aes192() {
        fips197_block(
            "000102030405060708090a0b0c0d0e0f1011121314151617",
            "dda97ca4864cdfe06eaf70a0ec0d7191",
        );
    }

    #[test]
    fn fips197_c3_aes256() {
        fips197_block(
            "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f",
            "8ea2b7ca516745bfeafc49904b496089",
        );
    }

    #[test]
    fn key_length_validation() {
        assert!(matches!(AesKey::from_bytes(&[0u8; 16]).unwrap(), AesKey::K128(_)));
        assert!(matches!(AesKey::from_bytes(&[0u8; 24]).unwrap(), AesKey::K192(_)));
        assert!(matches!(AesKey::from_bytes(&[0u8; 32]).unwrap(), AesKey::K256(_)));
        assert!(AesKey::from_bytes(&[0u8; 15]).is_err());
        assert!(AesKey::from_bytes(&[]).is_err());
        assert!(AesKey::from_bytes(&[0u8; 33]).is_err());
    }

    // ---- NIST SP 800-38A ----

    const NIST_128_KEY: &str = "2b7e151628aed2a6abf7158809cf4f3c";
    const NIST_PT: [&str; 4] = [
        "6bc1bee22e409f96e93d7e117393172a",
        "ae2d8a571e03ac9c9eb76fac45af8e51",
        "30c81c46a35ce411e5fbc1191a0a52ef",
        "f69f2445df4f9b17ad2b417be66c3710",
    ];

    fn nist_pt() -> Vec<u8> {
        NIST_PT.iter().flat_map(|b| hex(b)).collect()
    }

    /// F.1.1 ECB-AES128：4 块官方向量。
    #[test]
    fn nist_f11_ecb_aes128() {
        let ct: Vec<u8> = [
            "3ad77bb40d7a3660a89ecaf32466ef97",
            "f5d3d58503b9699de785895a96fdbaaf",
            "43b1cd7f598ece23881b00e3ed030688",
            "7b0c785e27e8ad3f8223207104725dd4",
        ]
        .iter()
        .flat_map(|b| hex(b))
        .collect();
        let key = AesKey::from_bytes(&hex(NIST_128_KEY)).unwrap();
        assert_eq!(aes_ecb_decrypt(&key, &ct).unwrap(), nist_pt());
        // 加密方向也用单块入口核对一遍，确认 encrypt_block 与官方一致
        for (i, pt_block) in NIST_PT.iter().enumerate() {
            let mut b = arr16(pt_block);
            key.encrypt_block(&mut b);
            assert_eq!(b.to_vec(), hex(&ct_hex_at(i)));
        }
    }

    fn ct_hex_at(i: usize) -> String {
        [
            "3ad77bb40d7a3660a89ecaf32466ef97",
            "f5d3d58503b9699de785895a96fdbaaf",
            "43b1cd7f598ece23881b00e3ed030688",
            "7b0c785e27e8ad3f8223207104725dd4",
        ][i]
        .to_string()
    }

    /// F.2.1 CBC-AES128：4 块官方向量。
    #[test]
    fn nist_f21_cbc_aes128() {
        let ct: Vec<u8> = [
            "7649abac8119b246cee98e9b12e9197d",
            "5086cb9b507219ee95db113a917678b2",
            "73bed6b8e3c1743b7116e69e22229516",
            "3ff1caa1681fac09120eca307586e1a7",
        ]
        .iter()
        .flat_map(|b| hex(b))
        .collect();
        let key = AesKey::from_bytes(&hex(NIST_128_KEY)).unwrap();
        let iv = arr16("000102030405060708090a0b0c0d0e0f");

        // 解密方向（裸，不去填充 —— 官方明文末字节恰好是 0x10，会被误当填充）
        assert_eq!(aes_cbc_decrypt_raw(&key, &iv, &ct).unwrap(), nist_pt());

        // 加密方向：用 encrypt_block 手搓 CBC 链接，结果必须等于官方向量。
        // 这一步同时验证了 encrypt_block 与链式 XOR 的正确性。
        let mut prev = iv;
        let mut got = Vec::new();
        for (i, pt_block) in NIST_PT.iter().enumerate() {
            let mut b = arr16(pt_block);
            for j in 0..16 {
                b[j] ^= prev[j];
            }
            key.encrypt_block(&mut b);
            assert_eq!(b.to_vec(), hex(&ct_hex_at_cbc(i)));
            prev = b;
            got.extend_from_slice(&b);
        }
        assert_eq!(got, ct);
    }

    fn ct_hex_at_cbc(i: usize) -> String {
        [
            "7649abac8119b246cee98e9b12e9197d",
            "5086cb9b507219ee95db113a917678b2",
            "73bed6b8e3c1743b7116e69e22229516",
            "3ff1caa1681fac09120eca307586e1a7",
        ][i]
        .to_string()
    }

    /// F.5.1 CTR-AES128：4 块官方向量。
    #[test]
    fn nist_f51_ctr_aes128() {
        let ct: Vec<u8> = [
            "874d6191b620e3261bef6864990db6ce",
            "9806f66b7970fdff8617187bb9fffdff",
            "5ae4df3edbd5d35e5b4f09020db03eab",
            "1e031dda2fbe03d1792170a0f3009cee",
        ]
        .iter()
        .flat_map(|b| hex(b))
        .collect();
        let key = AesKey::from_bytes(&hex(NIST_128_KEY)).unwrap();
        let iv = arr16("f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff");
        assert_eq!(aes_ctr_decrypt(&key, &iv, &ct).unwrap(), nist_pt());
        // CTR 自反：再异或一次回到密文
        assert_eq!(
            aes_ctr_decrypt(&key, &iv, &nist_pt()).unwrap(),
            ct,
            "CTR 加解密必须同构"
        );
    }

    /// CTR 计数器必须是 **128 位大端** 递增，不是只进最后一个字节。
    /// 用一块全 0xFF 的初始计数器来钉死这个行为：末字节进位要往前进位。
    #[test]
    fn ctr_counter_carries_across_bytes() {
        let key = AesKey::from_bytes(&hex(NIST_128_KEY)).unwrap();
        let iv = arr16("000000000000000000000000ffffffff");
        // 32 字节数据 → 两块，第二块的计数器应为 00000000000000000000000100000000
        let zero = [0u8; 32];
        let out = aes_ctr_decrypt(&key, &iv, &zero).unwrap();
        let mut c1 = iv;
        encrypt_block_with(&key.expand(), &mut c1);
        let mut c2 = arr16("00000000000000000000000100000000");
        encrypt_block_with(&key.expand(), &mut c2);
        assert_eq!(&out[..16], &c1[..]);
        assert_eq!(&out[16..], &c2[..]);
    }

    /// 非 16 整数倍的 CTR 输入要能正确处理（末块截断）。
    #[test]
    fn ctr_handles_partial_final_block() {
        let key = AesKey::from_bytes(&hex(NIST_128_KEY)).unwrap();
        let iv = arr16("f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff");
        let full = nist_pt();
        let ct = aes_ctr_decrypt(&key, &iv, &full).unwrap();
        let short_ct = &ct[..20];
        let got = aes_ctr_decrypt(&key, &iv, short_ct).unwrap();
        assert_eq!(got, full[..20].to_vec());
    }

    /// CBC 长度非 16 倍数要报错，不能"凑合解"。
    #[test]
    fn cbc_rejects_unaligned() {
        let key = AesKey::from_bytes(&hex(NIST_128_KEY)).unwrap();
        let iv = [0u8; 16];
        assert!(aes_cbc_decrypt(&key, &iv, &[0u8; 17]).is_err());
        assert!(aes_cbc_decrypt(&key, &iv, &[]).is_err());
        assert!(aes_ecb_decrypt(&key, &[0u8; 15]).is_err());
    }

    /// CBC 带 PKCS7 的往返：用 encrypt_block 手工构造密文（期望值就是明文本身）。
    #[test]
    fn cbc_pkcs7_roundtrip() {
        let key = AesKey::from_bytes(&hex(NIST_128_KEY)).unwrap();
        let iv = arr16("000102030405060708090a0b0c0d0e0f");
        let plain = b"media engine hls".to_vec(); // 16 字节，正好需要整块填充
        let mut padded = plain.clone();
        padded.extend_from_slice(&[16u8; 16]);
        // 手工 CBC 加密
        let mut prev = iv;
        let mut ct = Vec::new();
        for chunk in padded.chunks(16) {
            let mut b = [0u8; 16];
            b.copy_from_slice(chunk);
            for j in 0..16 {
                b[j] ^= prev[j];
            }
            key.encrypt_block(&mut b);
            prev = b;
            ct.extend_from_slice(&b);
        }
        assert_eq!(aes_cbc_decrypt(&key, &iv, &ct).unwrap(), plain);
    }

    /// 篡改密文 → 去填充失败必须报错，而不是返回被截断的"明文"。
    #[test]
    fn cbc_bad_padding_is_an_error_not_a_truncation() {
        let key = AesKey::from_bytes(&hex(NIST_128_KEY)).unwrap();
        let iv = arr16("000102030405060708090a0b0c0d0e0f");
        let ct: Vec<u8> = hex("7649abac8119b246cee98e9b12e9197d");
        // 官方向量末字节 0x7d 不是合法填充值 → 必须报错
        let e = aes_cbc_decrypt(&key, &iv, &ct).unwrap_err();
        assert!(e.to_string().contains("去填充失败"), "实际错误: {e}");
    }

    #[test]
    fn pkcs7_unpad_edges() {
        assert_eq!(pkcs7_unpad(&[1, 2, 3, 1]), Some(&[1u8, 2, 3][..]));
        assert_eq!(pkcs7_unpad(&[16u8; 16]), Some(&[][..]));
        assert_eq!(pkcs7_unpad(&[]), None);
        // pad = 0 非法
        assert_eq!(pkcs7_unpad(&[1, 2, 0]), None);
        // 填充字节不一致
        assert_eq!(pkcs7_unpad(&[1, 2, 3, 3]), None);
        // pad 超过数据长度
        assert_eq!(pkcs7_unpad(&[5]), None);
    }

    // ---- Base64 ----

    #[test]
    fn base64_rfc4648_vectors() {
        // RFC 4648 §10 官方样例（带填充）
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foob"), "Zm9vYg==");
        assert_eq!(base64_encode(b"fooba"), "Zm9vYmE=");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");

        assert_eq!(base64_decode("Zg==").unwrap(), b"f");
        assert_eq!(base64_decode("Zm8=").unwrap(), b"fo");
        assert_eq!(base64_decode("Zm9v").unwrap(), b"foo");
        assert_eq!(base64_decode("Zm9vYg==").unwrap(), b"foob");
        assert_eq!(base64_decode("Zm9vYmE=").unwrap(), b"fooba");
        assert_eq!(base64_decode("Zm9vYmFy").unwrap(), b"foobar");
    }

    #[test]
    fn base64_accepts_unpadded_and_urlsafe_and_whitespace() {
        // 无填充
        assert_eq!(base64_decode("Zg").unwrap(), b"f");
        assert_eq!(base64_decode("Zm8").unwrap(), b"fo");
        assert_eq!(base64_decode("Zm9vYg").unwrap(), b"foob");
        // URL-safe 字母表。选这三个字节是为了让标准字母表里同时出现 `+`(62) 与
        // `/`(63)，这样才能真的验证到"`-` `_` 与 `+` `/` 等价"这条映射；
        // 只用 `fb ef be` 的话四个字符全是 `+`，字母表映射根本没被覆盖。
        let raw: Vec<u8> = vec![0xfb, 0xff, 0xbe];
        assert_eq!(base64_encode(&raw), "+/++");
        assert_eq!(base64_decode("+/++").unwrap(), raw);
        assert_eq!(base64_decode("-_++").unwrap(), raw);
        // 换行/空格
        assert_eq!(base64_decode("Zm9v\nYmFy\n").unwrap(), b"foobar");
        // 非法字符与"填充后又来数据"
        assert!(base64_decode("Zm9v*").is_err());
        assert!(base64_decode("Zg==Zg").is_err());
    }

    #[test]
    fn base64_roundtrip_all_lengths() {
        for n in 0..64usize {
            let data: Vec<u8> = (0..n).map(|i| (i * 7 + 3) as u8).collect();
            assert_eq!(base64_decode(&base64_encode(&data)).unwrap(), data);
        }
    }

    // ---- 密钥体归一化 ----

    #[test]
    fn decode_key_body_three_forms() {
        let raw: Vec<u8> = (0..16u8).collect();
        // 1) 裸字节
        assert_eq!(decode_key_body(&raw, 16).unwrap(), raw);
        // 2) hex 文本（32 字符）
        let hex_text = "000102030405060708090a0b0c0d0e0f";
        assert_eq!(decode_key_body(hex_text.as_bytes(), 16).unwrap(), raw);
        // 大写 hex 也要认
        assert_eq!(
            decode_key_body(hex_text.to_uppercase().as_bytes(), 16).unwrap(),
            raw
        );
        // 3) Base64 文本（16 字节 → 24 字符，含 '=' 填充）
        let b64 = base64_encode(&raw);
        assert_eq!(b64.len(), 24);
        assert_eq!(decode_key_body(b64.as_bytes(), 16).unwrap(), raw);
        // 带尾随换行的 Base64（文本文件存密钥的常见形态）
        let with_nl = format!("{b64}\n");
        assert_eq!(decode_key_body(with_nl.as_bytes(), 16).unwrap(), raw);
    }

    #[test]
    fn decode_key_body_rejects_garbage() {
        assert!(decode_key_body(b"short", 16).is_err());
        // 24 字符但 base64 解出来不是 16 字节（"aaaa..." → 18 字节）
        assert!(decode_key_body(b"aaaaaaaaaaaaaaaaaaaaaaaa", 16).is_err());
    }
}
