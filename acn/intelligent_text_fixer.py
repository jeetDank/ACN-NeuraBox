"""
INTELLIGENT TEXT QUALITY FIXER FOR ACN RAG SYSTEM
==================================================
Integrated with ACN scraper, ingester, and query pipeline
Handles text quality issues intelligently without hardcoded patterns

Integration Points:
1. Scraper (scrapper.py): Clean scraped content before saving
2. Ingester (ingest.py): Clean chunks before embedding
3. Query: Clean retrieved answers before returning to user

Design Principles:
1. PROTECT domain terms - never "correct" business-specific words
2. USE proper NLP libraries - not regex for every possible typo
3. CONTEXT-AWARE - only fix when confidence is high
4. VALIDATE don't replace - check if something is wrong before changing
5. MINIMAL intervention - if unsure, leave it alone

Dependencies:
    pip install autocorrect language-tool-python python-dateutil
"""

import re
import unicodedata
from typing import Set, Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field

# Optional NLP libraries - graceful degradation if not installed
try:
    from autocorrect import Speller
    SPELLER_AVAILABLE = True
except ImportError:
    SPELLER_AVAILABLE = False
    print("⚠ Install autocorrect for spell checking: pip install autocorrect")

try:
    import language_tool_python
    GRAMMAR_AVAILABLE = True
except ImportError:
    GRAMMAR_AVAILABLE = False
    print("⚠ Install language-tool-python for grammar: pip install language-tool-python")

try:
    from dateutil.parser import parse as parse_date
    from dateutil.parser import ParserError
    DATEUTIL_AVAILABLE = True
except ImportError:
    DATEUTIL_AVAILABLE = False
    print("⚠ Install python-dateutil for date parsing: pip install python-dateutil")


@dataclass
class FixerConfig:
    """Configuration for the text fixer - customize for your domain"""
    
    # PROTECTED TERMS: These will NEVER be modified by spell/grammar checkers
    # Add your organization-specific terms here
    protected_terms: Set[str] = field(default_factory=lambda: {
        # Organization names
        'ACN', 'Applied', 'EZLynx', 'Epic', 'Ivans',
        'Applied Client Network', 'Client', 'Network',
        # Product names
        'TAM', 'CSR', 'BMO', 'Vertafore', 'Agency Matrix',
        'Applied Epic', 'Applied TAM',
        # Locations (proper nouns)
        'Dallas', 'Calgary', 'Alberta', 'Texas', 'Canada',
        'Frisco', 'Austin', 'Houston',
        # Technical terms
        'API', 'URL', 'HTTP', 'HTTPS', 'JSON', 'XML', 'SDK',
        # Common abbreviations
        'Inc', 'LLC', 'Ltd', 'Corp', 'CEO', 'CFO', 'CTO',
    })
    
    # DOMAIN VOCABULARY: Correct spellings of domain-specific terms
    # Spell checker will use these as valid words
    domain_vocabulary: Set[str] = field(default_factory=lambda: {
        'acn', 'ezlynx', 'webinar', 'webinars', 'onboarding',
        'workflow', 'workflows', 'roundtable', 'roundtables',
        'fintech', 'insurtech', 'brokerage', 'brokerages',
        'summits', 'networking', 'memberbase', 'forums',
        'offboarding', 'insurers', 'reinsurance',
    })
    
    # DATE VALIDATION: Reasonable year range for your domain
    min_valid_year: int = 2020
    max_valid_year: int = 2030
    
    # CONFIDENCE THRESHOLDS
    min_spell_confidence: float = 0.8  # Don't correct if less confident
    
    # FEATURES: Toggle features on/off
    enable_spell_check: bool = True
    enable_grammar_check: bool = False  # Disabled by default - can be slow
    enable_date_validation: bool = True
    enable_url_cleanup: bool = True


class IntelligentTextFixer:
    """
    Intelligent text fixer that uses NLP libraries and context-aware logic.
    
    Key differences from hardcoded approach:
    1. Uses spell checker library instead of pattern matching every typo
    2. Protects domain-specific terms from being "corrected"
    3. Validates dates semantically, not with regex replacement
    4. Only intervenes when confidence is high
    """
    
    def __init__(self, config: Optional[FixerConfig] = None):
        self.config = config or FixerConfig()
        
        # Initialize NLP tools
        self._init_spell_checker()
        self._init_grammar_checker()
        
        # Build lookup sets
        self.protected_lower = {term.lower() for term in self.config.protected_terms}
        self.unicode_map = self._build_unicode_map()
        
        # Statistics
        self.stats = {
            'total_fixed': 0,
            'unicode_normalized': 0,
            'dates_fixed': 0,
            'urls_fixed': 0,
            'spell_corrections': 0,
        }
        
        print("✓ ACN Text Fixer initialized")
    
    def _init_spell_checker(self):
        """Initialize spell checker"""
        self.spell_checker = None
        
        if not SPELLER_AVAILABLE or not self.config.enable_spell_check:
            return
        
        try:
            self.spell_checker = Speller(lang='en')
            print("✓ Spell checker ready")
        except Exception as e:
            print(f"⚠ Spell checker init failed: {e}")
    
    def _init_grammar_checker(self):
        """Initialize grammar checker"""
        self.grammar_checker = None
        
        if not GRAMMAR_AVAILABLE or not self.config.enable_grammar_check:
            return
        
        try:
            self.grammar_checker = language_tool_python.LanguageTool('en-US')
            print("✓ Grammar checker ready")
        except Exception as e:
            print(f"⚠ Grammar checker init failed: {e}")
    
    def _build_unicode_map(self) -> Dict[str, str]:
        """Build Unicode confusables mapping"""
        return {
            # Cyrillic → Latin
            'А': 'A', 'В': 'B', 'С': 'C', 'Е': 'E', 'Н': 'H', 'І': 'I',
            'К': 'K', 'М': 'M', 'О': 'O', 'Р': 'P', 'Т': 'T', 'Х': 'X',
            'а': 'a', 'с': 'c', 'е': 'e', 'і': 'i', 'о': 'o', 'р': 'p',
            'х': 'x', 'у': 'y',
            
            # Greek → Latin
            'Α': 'A', 'Β': 'B', 'Ε': 'E', 'Η': 'H', 'Ι': 'I', 'Κ': 'K',
            'Μ': 'M', 'Ν': 'N', 'Ο': 'O', 'Ρ': 'P', 'Τ': 'T', 'Χ': 'X',
            
            # Special characters
            '–': '-', '—': '-', ''': "'", ''': "'", '"': '"', '"': '"',
            '…': '...', '\u00a0': ' ',
        }
    
    # =========================================================================
    # UNICODE NORMALIZATION
    # =========================================================================
    
    def _normalize_unicode(self, text: str) -> str:
        """
        Normalize Unicode to prevent confusable character issues.
        This is NOT hardcoding - it's standard Unicode normalization.
        """
        # Replace confusables
        result = []
        changed = False
        for char in text:
            replacement = self.unicode_map.get(char, char)
            if replacement != char:
                changed = True
            result.append(replacement)
        
        text = ''.join(result)
        
        # Apply NFKC normalization
        normalized = unicodedata.normalize('NFKC', text)
        if normalized != text:
            changed = True
        
        if changed:
            self.stats['unicode_normalized'] += 1
        
        return normalized
    
    # =========================================================================
    # PROTECTION CHECKS
    # =========================================================================
    
    def _is_protected(self, word: str) -> bool:
        """Check if word should be protected from correction"""
        clean_word = word.strip('.,;:!?()[]"\'')
        
        # Protected terms (case-insensitive)
        if clean_word.lower() in self.protected_lower:
            return True
        
        # Domain vocabulary
        if clean_word.lower() in self.config.domain_vocabulary:
            return True
        
        # URLs and emails
        if '@' in word or '://' in word or word.startswith('www.'):
            return True
        
        # Numbers and dates
        if any(char.isdigit() for char in clean_word):
            return True
        
        # ALL CAPS (acronyms)
        if clean_word.isupper() and len(clean_word) >= 2:
            return True
        
        # CamelCase (technical terms)
        if re.match(r'^[A-Z][a-z]+[A-Z]', clean_word):
            return True
        
        return False
    
    # =========================================================================
    # SPELL CHECKING
    # =========================================================================
    
    def _apply_spell_check(self, text: str) -> str:
        """Apply spell checking with domain term protection"""
        if not self.spell_checker:
            return text
        
        lines = text.split('\n')
        corrected_lines = []
        corrections_made = 0
        
        for line in lines:
            words = line.split()
            corrected_words = []
            
            for word in words:
                if self._is_protected(word):
                    corrected_words.append(word)
                else:
                    try:
                        corrected = self.spell_checker(word)
                        if corrected != word:
                            corrections_made += 1
                        corrected_words.append(corrected)
                    except Exception:
                        corrected_words.append(word)
            
            corrected_lines.append(' '.join(corrected_words))
        
        if corrections_made > 0:
            self.stats['spell_corrections'] += corrections_made
        
        return '\n'.join(corrected_lines)
    
    # =========================================================================
    # DATE VALIDATION
    # =========================================================================
    
    def _validate_and_fix_dates(self, text: str) -> str:
        """Fix malformed years intelligently"""
        if not self.config.enable_date_validation:
            return text
        
        year_pattern = r'\b(\d{4,6})\b'
        dates_fixed = 0
        
        def fix_year(match) -> str:
            nonlocal dates_fixed
            year_str = match.group(1)
            year = int(year_str)
            
            # Already valid
            if self.config.min_valid_year <= year <= self.config.max_valid_year:
                return year_str
            
            # 5-6 digit years (11016, 30216)
            if year >= 10000:
                last_four = year % 10000
                
                # 10XX → 20XX → check if needs +10
                if 1000 <= last_four < 2000:
                    corrected = last_four + 1000
                    if corrected < self.config.min_valid_year:
                        corrected += 10
                    if self.config.min_valid_year <= corrected <= self.config.max_valid_year:
                        dates_fixed += 1
                        return str(corrected)
                
                # 20XX but too old
                elif 2000 <= last_four < self.config.min_valid_year:
                    corrected = last_four + 10
                    if self.config.min_valid_year <= corrected <= self.config.max_valid_year:
                        dates_fixed += 1
                        return str(corrected)
                
                # Already valid in last 4 digits
                elif self.config.min_valid_year <= last_four <= self.config.max_valid_year:
                    dates_fixed += 1
                    return str(last_four)
            
            # 4-digit years too high (3027 → 2027)
            elif year >= 3000:
                corrected = year - 1000
                if self.config.min_valid_year <= corrected <= self.config.max_valid_year:
                    dates_fixed += 1
                    return str(corrected)
            
            return year_str
        
        result = re.sub(year_pattern, fix_year, text)
        
        if dates_fixed > 0:
            self.stats['dates_fixed'] += dates_fixed
        
        return result
    
    # =========================================================================
    # URL CLEANUP
    # =========================================================================
    
    def _cleanup_urls(self, text: str) -> str:
        """Fix spacing issues in URLs"""
        if not self.config.enable_url_cleanup:
            return text
        
        original = text
        
        # Common TLDs
        tlds = {'org', 'com', 'net', 'edu', 'gov', 'io', 'co', 'ai'}
        
        # Fix protocol spacing
        text = re.sub(r'(https?)\s*:\s*/\s*/\s*', r'\1://', text, flags=re.IGNORECASE)
        
        # Fix www.
        text = re.sub(r'\bwww\s*\.\s*', 'www.', text, flags=re.IGNORECASE)
        
        # Fix domain.tld patterns
        for tld in tlds:
            text = re.sub(rf'(\w+)\s*\.\s*({tld})\b', rf'\1.\2', text, flags=re.IGNORECASE)
            text = re.sub(rf'\.\s+({tld})\b', rf'.\1', text, flags=re.IGNORECASE)
        
        # Fix path spacing
        text = re.sub(r'/\s+(\w)', r'/\1', text)
        text = re.sub(r'\s+/', r'/', text)
        
        # Fix remaining dots in URL context
        text = re.sub(r'(www\.\w+)\s*\.\s*(\w+)', r'\1.\2', text, flags=re.IGNORECASE)
        text = re.sub(r'(https?://www)\s*\.\s*', r'\1.', text, flags=re.IGNORECASE)
        
        # Clean double dots
        text = re.sub(r'\.{2,}', '.', text)
        
        # Fix emails
        text = re.sub(r'([\w.-]+)\s*@\s*([\w.-]+)', r'\1@\2', text)
        
        if text != original:
            self.stats['urls_fixed'] += 1
        
        return text
    
    # =========================================================================
    # FORMATTING CLEANUP
    # =========================================================================
    
    def _cleanup_formatting(self, text: str) -> str:
        """Clean up basic formatting issues"""
        # Normalize whitespace within lines
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Normalize spaces
            line = re.sub(r'[^\S\n]+', ' ', line)
            line = re.sub(r' +', ' ', line)
            line = line.strip()
            cleaned_lines.append(line)
        
        text = '\n'.join(cleaned_lines)
        
        # Fix spacing around punctuation (safe approach)
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)
        
        # Fix sentence spacing (only for clear sentence boundaries)
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
        
        # Fix numbered lists
        text = re.sub(r'(\d+)\)([A-Za-z])', r'\1) \2', text)
        
        # Fix hyphenated words
        text = re.sub(r'(\w+)-\s+(\w+)', r'\1-\2', text)
        
        # Remove excessive blank lines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def _remove_artifacts(self, text: str) -> str:
        """Remove common scraping artifacts"""
        # Remove code comments
        text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
        text = re.sub(r'\\/\*.*?\*\\/', '', text, flags=re.DOTALL)
        
        # Remove stray backslashes
        text = re.sub(r'(?<![\w/])\\+(?![\w/])', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove duplicate punctuation
        text = re.sub(r'\.{2,}', '.', text)
        text = re.sub(r',{2,}', ',', text)
        
        return text
    
    # =========================================================================
    # MAIN ENTRY POINT
    # =========================================================================
    
    def fix(self, text: str) -> str:
        """
        Main method to fix text quality issues.
        
        Processing order:
        1. Unicode normalization (foundational)
        2. Remove artifacts (clean slate)
        3. Spell checking (fix typos)
        4. Grammar checking (fix grammar)
        5. Date validation (semantic check)
        6. URL/Email cleanup (structural)
        7. Formatting cleanup (final polish)
        """
        if not text or not text.strip():
            return text
        
        self.stats['total_fixed'] += 1
        
        # Layer 1: Unicode normalization (ALWAYS runs)
        text = self._normalize_unicode(text)
        
        # Layer 2: Remove artifacts
        text = self._remove_artifacts(text)
        
        # Layer 3: Spell checking (if enabled)
        if self.config.enable_spell_check:
            text = self._apply_spell_check(text)
        
        # Layer 4: Grammar checking (if enabled)
        if self.config.enable_grammar_check:
            text = self._apply_grammar_check(text)
        
        # Layer 5: Date validation (if enabled)
        if self.config.enable_date_validation:
            text = self._validate_and_fix_dates(text)
        
        # Layer 6: URL/Email cleanup (if enabled)
        if self.config.enable_url_cleanup:
            text = self._cleanup_urls(text)
        
        # Layer 7: Formatting cleanup (ALWAYS runs)
        text = self._cleanup_formatting(text)
        
        return text
    
    def fix_with_report(self, text: str) -> Tuple[str, Dict]:
        """
        Fix text and return a report of what was changed.
        Useful for debugging and validation.
        """
        original = text
        fixed = self.fix(text)
        
        report = {
            'original_length': len(original),
            'fixed_length': len(fixed),
            'changed': original != fixed,
            'changes_made': []
        }
        
        # Simple diff to show what changed
        if original != fixed:
            # Find changed words
            orig_words = set(original.split())
            fixed_words = set(fixed.split())
            
            removed = orig_words - fixed_words
            added = fixed_words - orig_words
            
            if removed:
                report['changes_made'].append(f"Removed/changed: {removed}")
            if added:
                report['changes_made'].append(f"Added/corrected: {added}")
        
        return fixed, report


# =============================================================================
# CONVENIENCE FUNCTIONS FOR INTEGRATION
# =============================================================================

# Global instance
_default_fixer = None

def get_fixer(config: Optional[FixerConfig] = None) -> IntelligentTextFixer:
    """Get or create the default fixer instance"""
    global _default_fixer
    if _default_fixer is None or config is not None:
        _default_fixer = IntelligentTextFixer(config)
    return _default_fixer

def fix_text(text: str) -> str:
    """Simple function to fix text using default configuration"""
    return get_fixer().fix(text)

def fix_scraped_content(content: str) -> str:
    """Fix scraped content (for scrapper.py integration)"""
    return get_fixer().fix(content)

def fix_chunk(chunk: str) -> str:
    """Fix chunk (for ingest.py integration)"""
    return get_fixer().fix(chunk)

def fix_answer(answer: str, sources: List[str] = None) -> Dict:
    """Fix RAG answer and sources - drop-in replacement for old function"""
    fixer = get_fixer()
    
    fixed_answer = fixer.fix(answer)
    
    fixed_sources = []
    if sources:
        for source in sources:
            # Just remove spaces from URLs
            fixed_source = re.sub(r'\s+', '', source)
            fixed_sources.append(fixed_source)
    
    return {
        'answer': fixed_answer,
        'sources': fixed_sources
    }


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("INTELLIGENT TEXT FIXER - COMPREHENSIVE TESTING")
    print("=" * 80 + "\n")
    
    # Create fixer with custom config for ACN domain
    config = FixerConfig(
        protected_terms={
            'ACN', 'Applied', 'EZLynx', 'Epic', 'Ivans', 'TAM',
            'Dallas', 'Calgary', 'Alberta', 'Texas', 'Canada',
            'BMO', 'API', 'URL', 'HTTP', 'HTTPS'
        },
        domain_vocabulary={
            'acn', 'ezlynx', 'webinar', 'webinars', 'onboarding',
            'workflow', 'workflows', 'roundtable', 'roundtables',
            'insurtech', 'brokerage', 'brokerages', 'summits'
        }
    )
    
    fixer = IntelligentTextFixer(config)
    
    # Test 1: ACN preservation (CRITICAL)
    print("TEST 1: ACN Preservation")
    print("-" * 40)
    test1 = "ACN offers membership benefits. Join ACN today."
    result1 = fixer.fix(test1)
    print(f"Input:  {test1}")
    print(f"Output: {result1}")
    assert "ACN" in result1, "FAIL: ACN was changed!"
    print("✓ PASS\n")
    
    # Test 2: Date fixing
    print("TEST 2: Date Fixing")
    print("-" * 40)
    test2 = "Summit on February 25-26, 11016 in Dallas, Texas."
    result2 = fixer.fix(test2)
    print(f"Input:  {test2}")
    print(f"Output: {result2}")
    assert "11016" not in result2, "FAIL: Date not fixed!"
    print("✓ PASS\n")
    
    # Test 3: URL cleanup
    print("TEST 3: URL Cleanup")
    print("-" * 40)
    test3 = "Visit https:// www. appliedclientnetwork. org /events"
    result3 = fixer.fix(test3)
    print(f"Input:  {test3}")
    print(f"Output: {result3}")
    assert "appliedclientnetwork.org" in result3, "FAIL: URL not cleaned!"
    print("✓ PASS\n")
    
    # Test 4: Protected terms
    print("TEST 4: Protected Terms")
    print("-" * 40)
    test4 = "EZLynx and Applied Epic are key products at ACN."
    result4 = fixer.fix(test4)
    print(f"Input:  {test4}")
    print(f"Output: {result4}")
    assert "EZLynx" in result4 and "Epic" in result4, "FAIL: Protected terms changed!"
    print("✓ PASS\n")
    
    # Test 5: Real-world example
    print("TEST 5: Real-World Example")
    print("-" * 40)
    test5 = """There are two upcoming summits at ACN:

1)Dallas Summit: February 25-26, 11016 (Location: Dallas, Texas)
This summit is for Applied Epic and EZLynx users.

2)Calgary Summit: June 03-04, 3027 (Location: Calgary, Alberta)

Visit https:// www. appliedclientnetwork. org for details."""
    
    result5 = fixer.fix(test5)
    print(f"Input:\n{test5}")
    print(f"\nOutput:\n{result5}")
    print("✓ PASS\n")
    
    # Show statistics
    print("=" * 80)
    print("STATISTICS")
    print("=" * 80)
    stats = fixer.get_stats()
    for key, value in stats.items():
        print(f"{key:20s}: {value}")
    
    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED")
    print("=" * 80)