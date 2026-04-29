from pathlib import Path
from typing import List, Dict, Any

import cv2
from moviepy import VideoFileClip
from faster_whisper import WhisperModel

from Code.scripts.document_ocr import DocumentOCR


class VideoIngestor:
    def __init__(
        self,
        whisper_model: str = "base",
        device: str = "cuda",
        compute_type: str = "float16",
        frame_interval_sec: int = 10,
    ):
        self.asr = WhisperModel(
            whisper_model,
            device=device,
            compute_type=compute_type,
        )
        self.ocr = DocumentOCR(lang="en")
        self.frame_interval_sec = frame_interval_sec
        
    def extract_audio(self, video_path: str, out_dir: str = "data/processed/audio") -> str:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        audio_path = out / f"{Path(video_path).stem}.wav"

        clip = VideoFileClip(video_path)

        if clip.audio is None:
            clip.close()
            raise ValueError(f"No audio track found in video: {video_path}")

        clip.audio.write_audiofile(
            str(audio_path),
            codec="pcm_s16le",
            logger=None,
        )

        clip.close()

        return str(audio_path)

    def transcribe(self, audio_path: str, source_video: str) -> List[Dict[str, Any]]:
        segments, info = self.asr.transcribe(audio_path, beam_size=5)

        chunks = []
        for seg in segments:
            text = seg.text.strip()
            if not text:
                continue

            chunks.append({
                "source": source_video,
                "modality": "video_transcript",
                "start": seg.start,
                "end": seg.end,
                "text": text,
                "metadata": {
                    "language": info.language,
                    "asr_engine": "faster-whisper",
                }
            })

        return chunks

    def extract_frames(self, video_path: str, out_dir: str = "data/processed/video_frames") -> List[Dict[str, Any]]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        if not fps or fps <= 0:
            raise RuntimeError("Could not read video FPS.")

        interval_frames = int(fps * self.frame_interval_sec)
        frame_id = 0
        saved = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_id % interval_frames == 0:
                timestamp = frame_id / fps
                frame_path = out / f"{Path(video_path).stem}_{int(timestamp)}s.png"
                cv2.imwrite(str(frame_path), frame)

                saved.append({
                    "frame_path": str(frame_path),
                    "timestamp": timestamp,
                })

            frame_id += 1

        cap.release()
        return saved

    def ocr_frames(self, video_path: str) -> List[Dict[str, Any]]:
        frames = self.extract_frames(video_path)
        chunks = []

        for frame in frames:
            ocr_chunks = self.ocr.ocr_image(
                image_path=frame["frame_path"],
                source=video_path,
                page=None,
            )

            for chunk in ocr_chunks:
                chunk["modality"] = "video_frame_ocr"
                chunk["start"] = frame["timestamp"]
                chunk["end"] = frame["timestamp"] + self.frame_interval_sec
                chunk["metadata"]["timestamp"] = frame["timestamp"]

            chunks.extend(ocr_chunks)

        return chunks

    def process_video(self, video_path: str) -> List[Dict[str, Any]]:
        audio_path = self.extract_audio(video_path)

        transcript_chunks = self.transcribe(
            audio_path=audio_path,
            source_video=video_path,
        )

        frame_ocr_chunks = self.ocr_frames(video_path)

        return transcript_chunks + frame_ocr_chunks