"""
Steganography detection and prevention module for Sanctuary.
"""
import logging
import re
import math
import numpy as np
from typing import Dict, Any, List, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class StegDetector:
    def __init__(self):
        """Initialize steganography detection systems."""
        # Patterns are stored with their respective severity scores
        self.suspicious_patterns = [
            # Base encoding patterns
            ("Base64-Long", r'(?:[A-Za-z0-9+/]{16,})(?:={0,2})', 0.8),  # Base64 (long sequences)
            ("Base64-Medium", r'[A-Za-z0-9+/]{8,}={1,2}', 0.7),          # Base64 with padding (medium)
            ("Hex", r'(?:[0-9A-F]{2}){8,}', 0.6),                        # Hex encoding (longer sequences)
            
            # Unicode homoglyph patterns - High severity
            ("Cyrillic", r'[\u0430\u0435\u0456\u0458\u043E\u0460\u0455\u0441\u0443\u0445\u04CF\u04CE]', 0.8),  # Lookalikes
            ("ZeroWidth", r'[\u200B-\u200F\u202A-\u202E\uFEFF]', 0.9),  # Zero-width & directional characters
            ("Unicode", r'[\u2060-\u2069\u200C-\u200F]', 0.9),  # Joiners and invisible characters
            
            # Control character patterns - High severity
            ("Control", r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F]', 0.8),  # Control chars (excluding tab/newline)
            
            # Whitespace patterns - Low severity unless repeated
            ("TrailSpace", r'[ \t]{4,}$', 0.3),  # Excessive trailing whitespace
            ("EmptyLines", r'^\s{4,}$', 0.3),    # Excessive empty lines 
            ("ExcessSpace", r'\s{4,}', 0.3)      # Excessive spaces
        ]
        
        # Thresholds for detection
        self.base_entropy_threshold = 4.3      # Normal English text entropy
        self.suspicious_entropy_threshold = 5.0  # Threshold for suspicious content
        
    async def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze text content for potential steganographic content.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary containing analysis results and confidence scores
        """
        results = {
            "suspicious": False,
            "confidence": 0.0,
            "triggers": [],
            "encoding_entropy": 0.0
        }

        # 1. Check for suspicious patterns
        matches_found = []
        total_severity = 0.0
        
        for name, pattern, severity in self.suspicious_patterns:
            matches = list(re.finditer(pattern, text))
            for match in matches:
                # Skip matches that are likely legitimate
                if 'whitespace' in name.lower() and len(match.group()) < 4:
                    continue
                    
                matches_found.append({
                    "name": name,
                    "pattern": pattern,
                    "position": match.start(),
                    "match": match.group(),
                    "severity": severity
                })
                total_severity += severity
                
        results["triggers"].extend(matches_found)

        # 2. Calculate entropy measures
        results["encoding_entropy"] = self._calculate_entropy(text)
        
        # 3. Analyze statistical properties
        results["char_dist"] = self._analyze_char_distribution(text)
        
        # Start with assumption of safe content
        results["suspicious"] = False
        results["confidence"] = 0.0
        
        # 4. Make final determination based on multiple factors
        has_base64 = any("Base64" in p["name"] for p in matches_found)
        has_unicode = any("Unicode" in p["name"] for p in matches_found)
        
        # Check pattern matches first
        if total_severity >= 0.9:  # Single high severity pattern
            results["suspicious"] = True
            results["confidence"] = min(total_severity, 1.0)
        elif total_severity >= 0.7:  # Moderate patterns need corroboration
            if len(matches_found) > 1 or results["encoding_entropy"] > self.base_entropy_threshold:
                results["suspicious"] = True
                results["confidence"] = min(total_severity * 0.8, 1.0)
                
        # Handle entropy with context
        if results["encoding_entropy"] > self.suspicious_entropy_threshold:
            if len(matches_found) > 0:  # Any pattern match
                results["suspicious"] = True
                results["confidence"] = max(results["confidence"], 0.7)
            elif results["char_dist"]["special_ratio"] > 0.3 and results["char_dist"]["repetition_score"] < 0.4:
                # High entropy with non-repeating special chars
                results["suspicious"] = True
                results["confidence"] = 0.7
        
        # Special handling for Base64
        if has_base64 and results["encoding_entropy"] > self.base_entropy_threshold:
            results["suspicious"] = True
            results["confidence"] = max(results["confidence"], 0.7)
            
        return results

    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of the text."""
        if not text:
            return 0.0
            
        # Count character frequencies
        freq = {}
        for c in text:
            freq[c] = freq.get(c, 0) + 1
            
        # Calculate entropy
        length = len(text)
        entropy = 0.0
        for count in freq.values():
            p = count / length
            entropy -= p * math.log2(p)
            
        return entropy
        
    def _analyze_char_distribution(self, text: str) -> Dict[str, float]:
        """Analyze character distribution for anomalies."""
        dist = {
            "printable_ratio": 0.0,
            "space_ratio": 0.0,
            "special_ratio": 0.0,
            "repetition_score": 0.0
        }
        
        if not text:
            return dist
            
        length = len(text)
        printable = sum(1 for c in text if c.isprintable())
        spaces = sum(1 for c in text if c.isspace())
        special = sum(1 for c in text if not c.isalnum() and not c.isspace())
        
        # Calculate ratios
        dist["printable_ratio"] = printable / length
        dist["space_ratio"] = spaces / length
        dist["special_ratio"] = special / length
        
        # Calculate repetition score
        ngrams = [''.join(text[i:i+3]) for i in range(len(text)-2)]
        if ngrams:
            unique_ratio = len(set(ngrams)) / len(ngrams)
            dist["repetition_score"] = 1 - unique_ratio
            
        return dist

    async def verify_memory_block(self, block: Dict[str, Any]) -> bool:
        """
        Verify a memory block for potential steganographic content.
        
        Args:
            block: The memory block to verify
            
        Returns:
            True if block passes verification, False if suspicious
        """
        # Empty blocks are considered safe
        if not block:
            return True
            
        try:
            # 1. Track suspicious content count
            suspicious_count = 0
            high_confidence_count = 0
            
            # 2. Analyze all text fields
            for key, value in block.items():
                if isinstance(value, str):
                    result = await self.analyze_text(value)
                    if result["suspicious"]:
                        suspicious_count += 1
                        if result["confidence"] > 0.8:  # Higher threshold
                            high_confidence_count += 1
                        
                elif isinstance(value, (list, dict)):
                    # Recursively check nested structures
                    if not await self._verify_nested_content(value):
                        suspicious_count += 1
                        high_confidence_count += 1
            
            # 3. Make final determination
            # Only fail if we have multiple high confidence detections
            # or a very high confidence single detection
            return suspicious_count == 0 or (suspicious_count == 1 and high_confidence_count == 0)
            
        except Exception as e:
            logger.warning("Verification step failed, rejecting block: %s", e)
            return False
        
    # Maximum recursion depth for nested-content verification. Inputs to
    # this verifier can come from external sources (memory blocks, tool
    # outputs), so a deliberately-deep nesting would exhaust the Python
    # call stack and crash the cognitive loop. 32 is well past anything
    # legitimate memory blocks should produce.
    MAX_NESTED_DEPTH = 32

    async def _verify_nested_content(
        self, content: Union[List, Dict], _depth: int = 0
    ) -> bool:
        """Recursively verify nested content.

        Raises ValueError if nesting exceeds MAX_NESTED_DEPTH so an
        adversarial input can't recurse the call stack into oblivion.
        """
        if _depth >= self.MAX_NESTED_DEPTH:
            raise ValueError(
                f"Nested content exceeds MAX_NESTED_DEPTH={self.MAX_NESTED_DEPTH}; "
                f"refusing to recurse further (possible steganography attack)."
            )

        if isinstance(content, dict):
            for value in content.values():
                if isinstance(value, str):
                    result = await self.analyze_text(value)
                    if result["suspicious"] and result["confidence"] > 0.7:
                        return False
                elif isinstance(value, (list, dict)):
                    if not await self._verify_nested_content(
                        value, _depth=_depth + 1
                    ):
                        return False

        elif isinstance(content, list):
            for item in content:
                if isinstance(item, str):
                    result = await self.analyze_text(item)
                    if result["suspicious"] and result["confidence"] > 0.7:
                        return False
                elif isinstance(item, (list, dict)):
                    if not await self._verify_nested_content(
                        item, _depth=_depth + 1
                    ):
                        return False

        return True
        
    def _verify_block_structure(self, block: Dict[str, Any]) -> bool:
        """Verify the structural integrity of a memory block."""
        required_fields = {"content", "metadata", "timestamp"}
        
        # Check required fields
        if not all(field in block for field in required_fields):
            return False
            
        # Verify metadata structure
        if not isinstance(block["metadata"], dict):
            return False
            
        # Additional structure checks can be added here
        
        return True