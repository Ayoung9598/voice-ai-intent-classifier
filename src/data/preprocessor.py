"""
Text Preprocessing Pipeline for Voice AI Intent Classifier

Handles ASR noise, code-switching, and text normalization for 
multilingual (Kinyarwanda/English/Mixed) voice transcripts.
"""

import re
import unicodedata
from typing import Optional, List, Dict
from dataclasses import dataclass


@dataclass
class PreprocessingConfig:
    """Configuration for text preprocessing."""
    normalize_unicode: bool = True
    fix_repeated_chars: bool = True
    min_repeat_chars: int = 3
    lowercase: bool = False  # Keep case for transformer models
    preserve_code_switching: bool = True
    remove_extra_whitespace: bool = True


class TextPreprocessor:
    """
    Preprocessor for noisy ASR transcripts with code-switching support.
    
    Designed to handle:
    - Kinyarwanda/English/French mixed utterances
    - Common ASR transcription errors
    - Repeated characters from speech hesitation
    - Unicode normalization for Kinyarwanda diacritics
    """
    
    # Common ASR spelling errors mapping
    ASR_CORRECTIONS: Dict[str, str] = {
        # Common misspellings in the dataset
        "aplikasiyo": "application",
        "applicashon": "application",
        "aplication": "application",
        "sttatus": "status",
        "statuz": "status",
    }
    
    # Filler words to optionally remove
    FILLER_WORDS: List[str] = [
        "um", "uh", "hmm", "ahh", "err", "like",
    ]
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        Initialize the preprocessor.
        
        Args:
            config: Preprocessing configuration. Uses defaults if not provided.
        """
        self.config = config or PreprocessingConfig()
        
    def preprocess(self, text: str) -> str:
        """
        Apply full preprocessing pipeline to text.
        
        Args:
            text: Raw utterance text from ASR transcript
            
        Returns:
            Preprocessed text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Step 1: Unicode normalization (important for Kinyarwanda)
        if self.config.normalize_unicode:
            text = self._normalize_unicode(text)
        
        # Step 2: Fix repeated characters (e.g., "helllp" -> "help")
        if self.config.fix_repeated_chars:
            text = self._fix_repeated_chars(text)
        
        # Step 3: Light ASR error correction
        text = self._correct_asr_errors(text)
        
        # Step 4: Normalize whitespace
        if self.config.remove_extra_whitespace:
            text = self._normalize_whitespace(text)
        
        # Step 5: Optional lowercase (disabled by default for transformers)
        if self.config.lowercase:
            text = text.lower()
        
        return text.strip()
    
    def _normalize_unicode(self, text: str) -> str:
        """
        Normalize Unicode characters.
        
        Uses NFC normalization to handle Kinyarwanda diacritics consistently.
        """
        return unicodedata.normalize("NFC", text)
    
    def _fix_repeated_chars(self, text: str) -> str:
        """
        Reduce repeated characters to max 2 occurrences.
        
        Handles speech hesitation artifacts like "helllllp" -> "help"
        """
        # Pattern: 3+ of the same character -> 1 character
        # We use 1 instead of 2 because most words don't have double letters
        pattern = r'(.)\1{' + str(self.config.min_repeat_chars - 1) + r',}'
        return re.sub(pattern, r'\1', text)
    
    def _correct_asr_errors(self, text: str) -> str:
        """
        Apply light correction for common ASR transcription errors.
        
        Note: I am being conservative here to avoid over-correcting valid
        Kinyarwanda words that might look like misspellings.
        """
        words = text.split()
        corrected_words = []
        
        for word in words:
            # Check if lowercase version matches a known error
            word_lower = word.lower()
            if word_lower in self.ASR_CORRECTIONS:
                # Preserve original case pattern if possible
                corrected = self.ASR_CORRECTIONS[word_lower]
                if word[0].isupper() and len(corrected) > 0:
                    corrected = corrected[0].upper() + corrected[1:]
                corrected_words.append(corrected)
            else:
                corrected_words.append(word)
        
        return " ".join(corrected_words)
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize multiple whitespaces to single space."""
        return re.sub(r'\s+', ' ', text)
    
    def _remove_filler_words(self, text: str) -> str:
        """
        Remove common filler words.
        
        Note: Use with caution - fillers might carry meaning in some contexts.
        """
        words = text.split()
        filtered = [w for w in words if w.lower() not in self.FILLER_WORDS]
        return " ".join(filtered)
    
    def batch_preprocess(self, texts: List[str]) -> List[str]:
        """
        Preprocess a batch of texts.
        
        Args:
            texts: List of raw utterance texts
            
        Returns:
            List of preprocessed texts
        """
        return [self.preprocess(text) for text in texts]
    
    def get_text_stats(self, text: str) -> Dict:
        """
        Get statistics about the text for analysis.
        
        Args:
            text: Preprocessed or raw text
            
        Returns:
            Dictionary with text statistics
        """
        words = text.split() if text else []
        
        return {
            "char_count": len(text) if text else 0,
            "word_count": len(words),
            "avg_word_length": sum(len(w) for w in words) / len(words) if words else 0,
            "has_kinyarwanda_markers": self._has_kinyarwanda_patterns(text),
            "has_english_markers": self._has_english_patterns(text),
        }
    
    def _has_kinyarwanda_patterns(self, text: str) -> bool:
        """
        Detect if text likely contains Kinyarwanda.
        
        Uses common Kinyarwanda patterns/words as indicators.
        """
        if not text:
            return False
            
        kinyarwanda_patterns = [
            r'\bndashaka\b', r'\bnashaka\b', r'\bnkeneye\b',
            r'\bese\b', r'\bni\b', r'\bkuri\b', r'\byanjye\b',
            r'\bgusaba\b', r'\bgusubika\b', r'\bkumenya\b',
            r"y'", r"cy'", r"bw'",  # Kinyarwanda contractions
        ]
        
        text_lower = text.lower()
        return any(re.search(p, text_lower) for p in kinyarwanda_patterns)
    
    def _has_english_patterns(self, text: str) -> bool:
        """
        Detect if text likely contains English.
        """
        if not text:
            return False
            
        english_patterns = [
            r'\bthe\b', r'\bis\b', r'\bmy\b', r'\bwhat\b',
            r'\bhow\b', r'\bcan\b', r'\bhelp\b', r'\bwant\b',
            r'\bapplication\b', r'\bstatus\b', r'\bfee\b',
        ]
        
        text_lower = text.lower()
        return any(re.search(p, text_lower) for p in english_patterns)


def create_preprocessor(config_dict: Optional[Dict] = None) -> TextPreprocessor:
    """
    Factory function to create a preprocessor from config dictionary.
    
    Args:
        config_dict: Configuration dictionary (e.g., from YAML)
        
    Returns:
        Configured TextPreprocessor instance
    """
    if config_dict is None:
        return TextPreprocessor()
    
    config = PreprocessingConfig(
        normalize_unicode=config_dict.get("normalize_unicode", True),
        fix_repeated_chars=config_dict.get("fix_repeated_chars", True),
        min_repeat_chars=config_dict.get("min_repeat_chars", 3),
        lowercase=config_dict.get("lowercase", False),
        preserve_code_switching=config_dict.get("preserve_code_switching", True),
        remove_extra_whitespace=config_dict.get("remove_extra_whitespace", True),
    )
    
    return TextPreprocessor(config)


# Quick test
if __name__ == "__main__":
    preprocessor = TextPreprocessor()
    
    test_cases = [
        "Ndashaka kureba status ya application yanjye.",
        "I submitted my aplikasiyo but I can't see the sttatus.",
        "Helllllp me with my application pleaseeee",
        "Ese passport yanjye igeze he?",
    ]
    
    print("Preprocessing Examples:")
    print("-" * 60)
    for text in test_cases:
        processed = preprocessor.preprocess(text)
        stats = preprocessor.get_text_stats(processed)
        print(f"Original: {text}")
        print(f"Processed: {processed}")
        print(f"Stats: {stats}")
        print("-" * 60)
