from __future__ import annotations

from pathlib import Path

import cv2
from faster_whisper import WhisperModel
from moviepy import VideoFileClip

from Code.core.schema import RAGChunk
from Code.core.settings import (
    PROCESSED_DIR,
    VIDEO_FRAME_INTERVAL_SEC,
    WHISPER_COMPUTE_TYPE,
    WHISPER_DEVICE,
    WHISPER_MODEL_NAME,
)
from Code.ingestion.document_ocr import DocumentOCR


class VideoIngestor:
    """
    Video ingestion:
    - extracts audio and transcribes it with faster-whisper
    - samples frames and OCRs visual text with PaddleOCR
    """

    def __init__(
        self,
        whisper_model: str = WHISPER_MODEL_NAME,
        device: str = WHISPER_DEVICE,
        compute_type: str = WHISPER_COMPUTE_TYPE,
        frame_interval_sec: int = VIDEO_FRAME_INTERVAL_SEC,
        ocr: DocumentOCR | None = None,
    ):
        self.whisper_model = whisper_model
        self.device = device
        self.compute_type = compute_type
        self.frame_interval_sec = frame_interval_sec

        self.asr = WhisperModel(
            whisper_model,
            device=device,
            compute_type=compute_type,
        )

        self.ocr = ocr or DocumentOCR(lang="en")

    def extract_audio(
        self,
        video_path: str | Path,
        out_dir: str | Path | None = None,
    ) -> Path:
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        if out_dir is None:
            out_dir = PROCESSED_DIR / "audio" / video_path.stem

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        audio_path = out_dir / f"{video_path.stem}.wav"

        clip = VideoFileClip(str(video_path))

        try:
            if clip.audio is None:
                raise ValueError(f"No audio track found in video: {video_path}")

            clip.audio.write_audiofile(
                str(audio_path),
                codec="pcm_s16le",
                logger=None,
            )
        finally:
            clip.close()

        return audio_path

    def transcribe_audio(
        self,
        audio_path: str | Path,
        source_video: str | Path,
        chunk_start_id: int = 0,
    ) -> list[RAGChunk]:
        audio_path = Path(audio_path)
        source_video = Path(source_video)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        segments, info = self.asr.transcribe(
            str(audio_path),
            beam_size=5,
        )

        chunks: list[RAGChunk] = []

        for segment in segments:
            text = segment.text.strip()

            if not text:
                continue

            chunks.append(
                RAGChunk(
                    chunk_id=chunk_start_id + len(chunks),
                    text=text,
                    source=str(source_video),
                    source_name=source_video.name,
                    modality="video_transcript",
                    page=None,
                    block_type="speech",
                    start=float(segment.start),
                    end=float(segment.end),
                    metadata={
                        "asr_engine": "faster-whisper",
                        "whisper_model": self.whisper_model,
                        "language": getattr(info, "language", None),
                        "audio_path": str(audio_path),
                    },
                )
            )

        return chunks

    def extract_frames(
        self,
        video_path: str | Path,
        out_dir: str | Path | None = None,
    ) -> list[dict]:
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        if out_dir is None:
            out_dir = PROCESSED_DIR / "video_frames" / video_path.stem

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)

        if not fps or fps <= 0:
            cap.release()
            raise RuntimeError(f"Could not read FPS for video: {video_path}")

        interval_frames = max(1, int(fps * self.frame_interval_sec))

        frames: list[dict] = []
        frame_id = 0

        try:
            while True:
                ok, frame = cap.read()

                if not ok:
                    break

                if frame_id % interval_frames == 0:
                    timestamp = frame_id / fps
                    frame_path = out_dir / f"{video_path.stem}_{int(timestamp)}s.png"
                    cv2.imwrite(str(frame_path), frame)

                    frames.append(
                        {
                            "frame_path": frame_path,
                            "timestamp": float(timestamp),
                        }
                    )

                frame_id += 1
        finally:
            cap.release()

        return frames

    def ocr_frames(
        self,
        video_path: str | Path,
        chunk_start_id: int = 0,
    ) -> list[RAGChunk]:
        video_path = Path(video_path)
        frames = self.extract_frames(video_path)

        chunks: list[RAGChunk] = []

        for frame in frames:
            frame_chunks = self.ocr.ocr_image_file(
                image_path=frame["frame_path"],
                source=video_path,
                source_name=video_path.name,
                page=None,
                modality="video_frame_ocr",
                block_type="slide_text",
                chunk_id=chunk_start_id + len(chunks),
            )

            for chunk in frame_chunks:
                chunk.start = frame["timestamp"]
                chunk.end = frame["timestamp"] + self.frame_interval_sec
                chunk.metadata["timestamp"] = frame["timestamp"]
                chunk.metadata["frame_path"] = str(frame["frame_path"])

            chunks.extend(frame_chunks)

        for i, chunk in enumerate(chunks, start=chunk_start_id):
            chunk.chunk_id = i

        return chunks

    def process_video(
        self,
        video_path: str | Path,
        chunk_start_id: int = 0,
        include_frame_ocr: bool = True,
    ) -> list[RAGChunk]:
        video_path = Path(video_path)

        audio_path = self.extract_audio(video_path)

        transcript_chunks = self.transcribe_audio(
            audio_path=audio_path,
            source_video=video_path,
            chunk_start_id=chunk_start_id,
        )

        chunks = list(transcript_chunks)

        if include_frame_ocr:
            frame_chunks = self.ocr_frames(
                video_path=video_path,
                chunk_start_id=chunk_start_id + len(chunks),
            )
            chunks.extend(frame_chunks)

        for i, chunk in enumerate(chunks, start=chunk_start_id):
            chunk.chunk_id = i

        return chunks