"""
Speaker Alignment Module

Merges diarization and ASR outputs to produce speaker-attributed transcripts.
"""
from typing import List, Optional, Tuple
from dataclasses import dataclass

from ..diarization.pyannote_diarizer import SpeakerSegment
from ..transcription.whisper_asr import TranscriptSegment


@dataclass
class AlignedSegment:
    """Container for aligned transcript with speaker information."""
    speaker_id: str           # Speaker identifier
    speaker_index: int        # Numeric speaker index
    text: str                 # Transcribed text
    start_time: float         # Start time in seconds
    end_time: float           # End time in seconds
    original_text: str = ""   # Original (untranslated) text
    translated_text: str = "" # Translated text
    
    @property
    def duration(self) -> float:
        """Segment duration in seconds."""
        return self.end_time - self.start_time
    
    def format_timestamp(self) -> str:
        """Format start time as MM:SS."""
        minutes = int(self.start_time // 60)
        seconds = int(self.start_time % 60)
        return f"{minutes:02d}:{seconds:02d}"


class SpeakerAlignment:
    """
    Aligns ASR transcripts with speaker diarization.
    
    Matches transcript segments to speaker segments based on temporal overlap.
    
    Usage:
        alignment = SpeakerAlignment()
        aligned = alignment.align(transcript_segments, speaker_segments)
        for seg in aligned:
            print(f"[{seg.speaker_id}] {seg.text}")
    """
    
    def __init__(
        self,
        overlap_threshold: float = 0.5,  # Minimum overlap ratio to assign speaker
        default_speaker: str = "Speaker_1",
    ):
        """
        Initialize speaker alignment.
        
        Args:
            overlap_threshold: Minimum overlap ratio to confidently assign speaker
            default_speaker: Default speaker ID when no match found
        """
        self.overlap_threshold = overlap_threshold
        self.default_speaker = default_speaker
    
    def _calculate_overlap(
        self,
        seg1_start: float,
        seg1_end: float,
        seg2_start: float,
        seg2_end: float,
    ) -> Tuple[float, float]:
        """
        Calculate overlap between two time segments.
        
        Returns:
            Tuple of (overlap_duration, overlap_ratio relative to seg1)
        """
        overlap_start = max(seg1_start, seg2_start)
        overlap_end = min(seg1_end, seg2_end)
        overlap_duration = max(0, overlap_end - overlap_start)
        
        seg1_duration = seg1_end - seg1_start
        if seg1_duration > 0:
            overlap_ratio = overlap_duration / seg1_duration
        else:
            overlap_ratio = 0.0
        
        return overlap_duration, overlap_ratio
    
    def find_speaker_for_segment(
        self,
        transcript_seg: TranscriptSegment,
        speaker_segments: List[SpeakerSegment],
    ) -> Tuple[str, int]:
        """
        Find the best matching speaker for a transcript segment.
        
        Args:
            transcript_seg: Transcript segment to match
            speaker_segments: List of speaker segments
        
        Returns:
            Tuple of (speaker_id, speaker_index)
        """
        if not speaker_segments:
            return self.default_speaker, 0
        
        best_speaker_id = self.default_speaker
        best_speaker_index = 0
        best_overlap = 0.0
        
        for speaker_seg in speaker_segments:
            overlap_duration, overlap_ratio = self._calculate_overlap(
                transcript_seg.start_time,
                transcript_seg.end_time,
                speaker_seg.start_time,
                speaker_seg.end_time,
            )
            
            if overlap_duration > best_overlap:
                best_overlap = overlap_duration
                best_speaker_id = speaker_seg.speaker_id
                best_speaker_index = speaker_seg.speaker_index
        
        return best_speaker_id, best_speaker_index
    
    def align(
        self,
        transcript_segments: List[TranscriptSegment],
        speaker_segments: List[SpeakerSegment],
    ) -> List[AlignedSegment]:
        """
        Align transcript segments with speaker segments.
        
        Args:
            transcript_segments: ASR output segments
            speaker_segments: Diarization output segments
        
        Returns:
            List of AlignedSegment objects with speaker attribution
        """
        aligned = []
        
        for transcript_seg in transcript_segments:
            speaker_id, speaker_index = self.find_speaker_for_segment(
                transcript_seg, speaker_segments
            )
            
            aligned_seg = AlignedSegment(
                speaker_id=speaker_id,
                speaker_index=speaker_index,
                text=transcript_seg.text,
                start_time=transcript_seg.start_time,
                end_time=transcript_seg.end_time,
                original_text=transcript_seg.text,
            )
            aligned.append(aligned_seg)
        
        return aligned
    
    def align_and_merge(
        self,
        transcript_segments: List[TranscriptSegment],
        speaker_segments: List[SpeakerSegment],
        merge_gap: float = 1.0,  # Merge segments from same speaker within this gap
    ) -> List[AlignedSegment]:
        """
        Align and merge consecutive segments from the same speaker.
        
        Args:
            transcript_segments: ASR output segments
            speaker_segments: Diarization output segments
            merge_gap: Maximum gap in seconds to merge segments
        
        Returns:
            List of merged AlignedSegment objects
        """
        aligned = self.align(transcript_segments, speaker_segments)
        
        if len(aligned) <= 1:
            return aligned
        
        merged = []
        current = aligned[0]
        
        for next_seg in aligned[1:]:
            # Check if should merge with current
            same_speaker = current.speaker_id == next_seg.speaker_id
            small_gap = (next_seg.start_time - current.end_time) <= merge_gap
            
            if same_speaker and small_gap:
                # Merge: extend current segment
                current = AlignedSegment(
                    speaker_id=current.speaker_id,
                    speaker_index=current.speaker_index,
                    text=f"{current.text} {next_seg.text}",
                    start_time=current.start_time,
                    end_time=next_seg.end_time,
                    original_text=f"{current.original_text} {next_seg.original_text}",
                    translated_text=f"{current.translated_text} {next_seg.translated_text}".strip(),
                )
            else:
                # Different speaker or large gap: save current, start new
                merged.append(current)
                current = next_seg
        
        # Don't forget the last segment
        merged.append(current)
        
        return merged
    
    def split_by_speaker_change(
        self,
        text: str,
        start_time: float,
        end_time: float,
        speaker_segments: List[SpeakerSegment],
    ) -> List[AlignedSegment]:
        """
        Split a single text by speaker changes.
        
        This is useful when ASR doesn't provide word-level timestamps
        but we have speaker change points from diarization.
        
        Args:
            text: Full text to split
            start_time: Start time of text
            end_time: End time of text
            speaker_segments: Speaker segments with change points
        
        Returns:
            List of AlignedSegment objects split by speaker
        """
        if not speaker_segments:
            return [
                AlignedSegment(
                    speaker_id=self.default_speaker,
                    speaker_index=0,
                    text=text,
                    start_time=start_time,
                    end_time=end_time,
                    original_text=text,
                )
            ]
        
        # Find speaker segments that overlap with our time range
        relevant_segments = []
        for seg in speaker_segments:
            overlap, _ = self._calculate_overlap(start_time, end_time, seg.start_time, seg.end_time)
            if overlap > 0:
                relevant_segments.append(seg)
        
        if not relevant_segments:
            return [
                AlignedSegment(
                    speaker_id=self.default_speaker,
                    speaker_index=0,
                    text=text,
                    start_time=start_time,
                    end_time=end_time,
                    original_text=text,
                )
            ]
        
        # If only one speaker, no split needed
        if len(relevant_segments) == 1:
            seg = relevant_segments[0]
            return [
                AlignedSegment(
                    speaker_id=seg.speaker_id,
                    speaker_index=seg.speaker_index,
                    text=text,
                    start_time=start_time,
                    end_time=end_time,
                    original_text=text,
                )
            ]
        
        # Multiple speakers: split text proportionally by time
        # This is an approximation since we don't have word timestamps
        text_duration = end_time - start_time
        words = text.split()
        total_words = len(words)
        
        if total_words == 0:
            return []
        
        results = []
        word_index = 0
        
        for seg in relevant_segments:
            # Calculate what portion of text this speaker covers
            seg_start = max(seg.start_time, start_time)
            seg_end = min(seg.end_time, end_time)
            seg_duration = seg_end - seg_start
            
            if seg_duration <= 0:
                continue
            
            # Estimate word count for this speaker
            word_ratio = seg_duration / text_duration
            word_count = max(1, int(total_words * word_ratio))
            
            # Get words for this speaker
            speaker_words = words[word_index:word_index + word_count]
            word_index += word_count
            
            if speaker_words:
                speaker_text = " ".join(speaker_words)
                results.append(
                    AlignedSegment(
                        speaker_id=seg.speaker_id,
                        speaker_index=seg.speaker_index,
                        text=speaker_text,
                        start_time=seg_start,
                        end_time=seg_end,
                        original_text=speaker_text,
                    )
                )
        
        # Handle any remaining words
        if word_index < total_words:
            remaining_words = words[word_index:]
            if remaining_words and results:
                # Add to last segment
                last = results[-1]
                results[-1] = AlignedSegment(
                    speaker_id=last.speaker_id,
                    speaker_index=last.speaker_index,
                    text=f"{last.text} {' '.join(remaining_words)}",
                    start_time=last.start_time,
                    end_time=last.end_time,
                    original_text=f"{last.original_text} {' '.join(remaining_words)}",
                )
        
        return results
