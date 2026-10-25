//! 把 `me-stream` 打开的结果收敛成"一条字节流 + 需要告知宿主的备注"。
//!
//! `stream` 与 `decode` 都要这一段；它只依赖 me-stream，不依赖任何可选模块，
//! 所以放在独立文件里 —— 纯音频构建不会因为用不到 fMP4 而多编一份 stream。

use std::sync::Arc;

use me_core::ReadAt;
use me_stream::OpenedSource;

pub(crate) fn pick_source(opened: OpenedSource) -> me_core::Result<(Arc<dyn ReadAt>, Option<String>)> {
    match opened {
        OpenedSource::Single(s) => Ok((s, None)),
        OpenedSource::Hls(h) => {
            let note = if h.is_live {
                Some("直播流：本版按当前快照播放，不做时移".to_string())
            } else {
                None
            };
            Ok((h.source, note))
        }
        OpenedSource::DashTracks(list) => {
            // DASH 的视频与音频是两条独立字节流，播放窗一次只播一条。
            if list.len() == 1 {
                let t = list.into_iter().next().expect("刚判过");
                return Ok((t.source, None));
            }
            let has_video = list.iter().any(|t| t.content_type == "video");
            let note = format!(
                "这份 DASH 有 {} 条轨（视频与音频分开），本流只含{}；要带声音请先用 mux 合并",
                list.len(),
                if has_video { "视频" } else { "音频" }
            );
            // 有画面优先：画面没声音还能看，只有声音算怎么回事
            let pick = list.iter().position(|t| t.content_type == "video").unwrap_or(0);
            let mut list = list;
            let t = list.swap_remove(pick);
            Ok((t.source, Some(note)))
        }
    }
}
