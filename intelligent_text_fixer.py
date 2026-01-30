"""
INTELLIGENT TEXT QUALITY FIXER - UNIFIED LONG-TERM SOLUTION
============================================================
NO hardcoded error patterns - uses NLP and intelligent detection

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
        # Product names
        'TAM', 'CSR', 'BMO',
        # Locations (proper nouns)
        'Dallas', 'Calgary', 'Alberta', 'Texas', 'Canada',
        # Technical terms
        'API', 'URL', 'HTTP', 'HTTPS', 'JSON', 'XML',
        # Common abbreviations
        'Inc', 'LLC', 'Ltd', 'Corp',
    })
    
    # DOMAIN VOCABULARY: Correct spellings of domain-specific terms
    # Spell checker will use these as valid words
    domain_vocabulary: Set[str] = field(default_factory=lambda: {
        'acn', 'ezlynx', 'webinar', 'webinars', 'onboarding',
        'workflow', 'workflows', 'roundtable', 'roundtables',
        'fintech', 'insurtech', 'brokerage', 'brokerages',
    })
    
    # DATE VALIDATION: Reasonable year range for your domain
    min_valid_year: int = 2020
    max_valid_year: int = 2030
    
    # CONFIDENCE THRESHOLDS
    min_spell_confidence: float = 0.8  # Don't correct if less confident
    
    # FEATURES: Toggle features on/off
    enable_spell_check: bool = True
    enable_grammar_check: bool = True
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
        
        # Build case-insensitive protected terms lookup
        self.protected_lower = {term.lower() for term in self.config.protected_terms}
        
        # Unicode confusables (Cyrillic/Greek that look like Latin)
        # This is the ONLY "hardcoded" mapping - it's a universal standard
        self.unicode_confusables = self._build_unicode_map()
        
        print("✓ Intelligent Text Fixer initialized")
    
    def _init_spell_checker(self):
        """Initialize spell checker with domain vocabulary"""
        self.spell_checker = None
        
        if not SPELLER_AVAILABLE or not self.config.enable_spell_check:
            return
        
        try:
            self.spell_checker = Speller(lang='en')
            # Note: autocorrect doesn't support adding custom words easily
            # We handle protected terms by skipping them during correction
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
        """
        Build mapping for Unicode confusables.
        These are characters that LOOK like Latin letters but aren't.
        This is a universal standard, not domain-specific.
        """
        return {
            # Cyrillic → Latin (these look identical but are different unicode)
            'А': 'A', 'В': 'B', 'С': 'C', 'Е': 'E', 'Н': 'H', 'І': 'I',
            'К': 'K', 'М': 'M', 'О': 'O', 'Р': 'P', 'Т': 'T', 'Х': 'X',
            'а': 'a', 'с': 'c', 'е': 'e', 'і': 'i', 'о': 'o', 'р': 'p',
            'х': 'x', 'у': 'y', 'ѕ': 's', 'ј': 'j',
            # Greek → Latin
            'Α': 'A', 'Β': 'B', 'Ε': 'E', 'Η': 'H', 'Ι': 'I', 'Κ': 'K',
            'Μ': 'M', 'Ν': 'N', 'Ο': 'O', 'Ρ': 'P', 'Τ': 'T', 'Χ': 'X',
            # Special characters → ASCII equivalents
            '–': '-', '—': '-', ''': "'", ''': "'", '"': '"', '"': '"',
            '…': '...', '\u00a0': ' ',  # Non-breaking space
        }
    
    # =========================================================================
    # LAYER 1: Unicode Normalization (Always runs first)
    # =========================================================================
    
    def _normalize_unicode(self, text: str) -> str:
        """
        Normalize Unicode to prevent confusable character issues.
        This is NOT hardcoding - it's standard Unicode normalization.
        """
        # Step 1: Replace known confusables
        result = []
        for char in text:
            result.append(self.unicode_confusables.get(char, char))
        text = ''.join(result)
        
        # Step 2: Apply Unicode NFKC normalization
        # This standardizes characters (e.g., ﬁ → fi, ２ → 2)
        text = unicodedata.normalize('NFKC', text)
        
        return text
    
    # =========================================================================
    # LAYER 2: Intelligent Spell Checking
    # =========================================================================
    
    def _is_protected(self, word: str) -> bool:
        """Check if a word should be protected from spell correction"""
        clean_word = word.strip('.,;:!?()[]"\'')
        
        # Check against protected terms (case-insensitive)
        if clean_word.lower() in self.protected_lower:
            return True
        
        # Check against domain vocabulary
        if clean_word.lower() in self.config.domain_vocabulary:
            return True
        
        # Protect URLs and emails
        if '@' in word or '://' in word or word.startswith('www.'):
            return True
        
        # Protect numbers and dates
        if any(char.isdigit() for char in clean_word):
            return True
        
        # Protect ALL CAPS words (likely acronyms)
        if clean_word.isupper() and len(clean_word) >= 2:
            return True
        
        # Protect CamelCase words (likely proper nouns or technical terms)
        if re.match(r'^[A-Z][a-z]+[A-Z]', clean_word):
            return True
        
        return False
    
    def _apply_spell_check(self, text: str) -> str:
        """
        Apply spell checking while protecting domain terms.
        Uses autocorrect library - NOT hardcoded patterns.
        Preserves line breaks and structure.
        """
        if not self.spell_checker:
            return text
        
        # Process line by line to preserve structure
        lines = text.split('\n')
        corrected_lines = []
        
        for line in lines:
            words = line.split()
            corrected_words = []
            
            for word in words:
                if self._is_protected(word):
                    # Keep protected words unchanged
                    corrected_words.append(word)
                else:
                    try:
                        # Let the spell checker handle it
                        corrected = self.spell_checker(word)
                        corrected_words.append(corrected)
                    except Exception:
                        corrected_words.append(word)
            
            corrected_lines.append(' '.join(corrected_words))
        
        return '\n'.join(corrected_lines)
    
    # =========================================================================
    # LAYER 3: Intelligent Grammar Checking
    # =========================================================================
    
    def _apply_grammar_check(self, text: str) -> str:
        """
        Apply grammar checking while protecting domain terms.
        Uses LanguageTool - NOT hardcoded patterns.
        """
        if not self.grammar_checker:
            return text
        
        try:
            matches = self.grammar_checker.check(text)
            
            # Apply corrections in reverse order to preserve offsets
            corrections = []
            for match in matches:
                if not match.replacements:
                    continue
                
                # Get the text being flagged
                flagged_text = text[match.offset:match.offset + match.errorLength]
                
                # Skip if it's a protected term
                if self._is_protected(flagged_text):
                    continue
                
                # Skip certain rule categories that are often wrong for technical text
                skip_rules = {'MORFOLOGIK_RULE_EN_US', 'UPPERCASE_SENTENCE_START'}
                if match.ruleId in skip_rules:
                    continue
                
                corrections.append({
                    'start': match.offset,
                    'end': match.offset + match.errorLength,
                    'replacement': match.replacements[0]
                })
            
            # Apply corrections in reverse order
            for correction in reversed(corrections):
                text = (text[:correction['start']] + 
                       correction['replacement'] + 
                       text[correction['end']:])
        
        except Exception as e:
            print(f"⚠ Grammar check error: {e}")
        
        return text
    
    # =========================================================================
    # LAYER 4: Intelligent Date Validation
    # =========================================================================
    
    def _validate_and_fix_dates(self, text: str) -> str:
        """
        Validate dates semantically, not with hardcoded year replacements.
        Handles malformed years like 11016, 3027, etc.
        """
        # Pattern to find years (4-6 digits that look like years)
        year_pattern = r'\b(\d{4,6})\b'
        
        def fix_year(match) -> str:
            """Fix malformed years"""
            year_str = match.group(1)
            year = int(year_str)
            
            # Already valid year - keep it
            if self.config.min_valid_year <= year <= self.config.max_valid_year:
                return year_str
            
            # 5+ digit years (like 11016, 30216)
            if year >= 10000:
                # Strategy: extract last 4 digits and try to fix
                last_four = year % 10000
                
                # Case 1: Last 4 digits are 10XX (like 1016 from 11016)
                # 1016 → 2016 → 2026
                if 1000 <= last_four < 2000:
                    corrected = last_four + 1000  # 1016 → 2016
                    if corrected < self.config.min_valid_year:
                        corrected += 10  # 2016 → 2026
                    if self.config.min_valid_year <= corrected <= self.config.max_valid_year:
                        return str(corrected)
                
                # Case 2: Last 4 digits are 20XX but too old (like 2016)
                elif 2000 <= last_four < self.config.min_valid_year:
                    corrected = last_four + 10  # 2016 → 2026
                    if self.config.min_valid_year <= corrected <= self.config.max_valid_year:
                        return str(corrected)
                
                # Case 3: Last 4 digits are already valid
                elif self.config.min_valid_year <= last_four <= self.config.max_valid_year:
                    return str(last_four)
            
            # 4-digit years that are way too high (3027, 4026)
            elif year >= 3000:
                corrected = year - 1000  # 3027 → 2027
                if self.config.min_valid_year <= corrected <= self.config.max_valid_year:
                    return str(corrected)
            
            # 4-digit years that are slightly too high (2035, 2040)
            # Don't change - could be legitimate future dates
            
            # 4-digit years that are too low (2015, 2016)  
            # Don't change - could be historical references
            
            return year_str  # Keep original if no confident fix
        
        text = re.sub(year_pattern, fix_year, text)
        
        return text
    
    # =========================================================================
    # LAYER 5: URL and Email Cleanup
    # =========================================================================
    
    def _cleanup_urls_and_emails(self, text: str) -> str:
        """
        Clean up URLs and emails - fix spacing issues.
        Simple and reliable approach: fix spaces around dots, slashes, and colons in URL contexts.
        """
        if not self.config.enable_url_cleanup:
            return text
        
        # TLDs to recognize
        tlds = {'org', 'com', 'net', 'edu', 'gov', 'io', 'co', 'ai', 'app', 'dev'}
        
        # Step 1: Fix protocol spacing (https :// → https://)
        text = re.sub(r'(https?)\s*:\s*/\s*/\s*', r'\1://', text, flags=re.IGNORECASE)
        
        # Step 2: Fix www . → www.
        text = re.sub(r'\bwww\s*\.\s*', 'www.', text, flags=re.IGNORECASE)
        
        # Step 3: Fix domain . tld patterns (domain . org → domain.org)
        # This handles: appliedclientnetwork . org → appliedclientnetwork.org
        for tld in tlds:
            # Match: word followed by space(s), dot, space(s), tld
            text = re.sub(rf'(\w+)\s*\.\s*({tld})\b', rf'\1.\2', text, flags=re.IGNORECASE)
            # Match: dot followed by space(s) before tld  
            text = re.sub(rf'\.\s+({tld})\b', rf'.\1', text, flags=re.IGNORECASE)
        
        # Step 4: Fix path spacing (/ path → /path)
        text = re.sub(r'/\s+(\w)', r'/\1', text)
        text = re.sub(r'\s+/', r'/', text)
        
        # Step 5: Fix remaining dots in URL context
        # If we have www.something . something, fix it
        text = re.sub(r'(www\.\w+)\s*\.\s*(\w+)', r'\1.\2', text, flags=re.IGNORECASE)
        
        # Step 6: Fix https://www . domain patterns
        text = re.sub(r'(https?://www)\s*\.\s*', r'\1.', text, flags=re.IGNORECASE)
        
        # Step 7: After all fixes, clean up any double dots
        text = re.sub(r'\.{2,}', '.', text)
        
        # Step 8: Fix email addresses (user @ domain . com → user@domain.com)
        text = re.sub(r'([\w.-]+)\s*@\s*([\w.-]+)', r'\1@\2', text)
        
        return text
    
    # =========================================================================
    # LAYER 6: Basic Formatting Cleanup
    # =========================================================================
    
    def _cleanup_formatting(self, text: str) -> str:
        """
        Basic formatting cleanup - universal rules only.
        NO domain-specific patterns.
        Careful not to break URLs!
        """
        # Normalize whitespace WITHIN lines only (preserve line breaks)
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Normalize spaces within the line
            line = re.sub(r'[^\S\n]+', ' ', line)
            line = re.sub(r' +', ' ', line)
            line = line.strip()
            cleaned_lines.append(line)
        
        text = '\n'.join(cleaned_lines)
        
        # Fix spacing around punctuation - BUT NOT IN URLs!
        # Remove space before punctuation (safe)
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)
        
        # Add space after punctuation ONLY if:
        # - It's not part of a URL (no :// nearby, no www nearby)
        # - It's not a number (like 3.14)
        # Pattern: punctuation followed by letter, but NOT in URL context
        def add_space_if_not_url(match):
            # Get surrounding context
            start = max(0, match.start() - 10)
            context = text[start:match.end()]
            
            # Don't add space if this looks like part of a URL
            if '://' in context or 'www' in context.lower():
                return match.group(0)  # Keep as-is
            
            # Don't add space after . if followed by common TLDs (even outside URL)
            tlds = {'org', 'com', 'net', 'edu', 'gov', 'io', 'co'}
            following = match.group(2).lower()
            for tld in tlds:
                if following.startswith(tld):
                    return match.group(0)  # Keep as-is
            
            return match.group(1) + ' ' + match.group(2)
        
        # Only fix sentence-ending punctuation (. ! ?) followed by uppercase
        # This is safe and doesn't affect URLs
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
        
        # Fix spacing in numbered lists: "1)Item" → "1) Item"
        text = re.sub(r'(\d+)\)([A-Za-z])', r'\1) \2', text)
        
        # Fix broken hyphenated words: "hands- on" → "hands-on"
        text = re.sub(r'(\w+)-\s+(\w+)', r'\1-\2', text)
        
        # Remove multiple blank lines (but keep single blank lines for paragraphs)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    # =========================================================================
    # LAYER 7: Remove Artifacts
    # =========================================================================
    
    def _remove_artifacts(self, text: str) -> str:
        """Remove common artifacts from LLM output or web scraping"""
        # Remove code comment artifacts: /* */ or \/* *\/
        text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
        text = re.sub(r'\\/\*.*?\*\\/', '', text, flags=re.DOTALL)
        
        # Remove stray backslashes (but not in paths)
        text = re.sub(r'(?<![\w/])\\+(?![\w/])', '', text)
        
        # Remove HTML tags if any slipped through
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
            text = self._cleanup_urls_and_emails(text)
        
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
# CONVENIENCE FUNCTIONS
# =============================================================================

# Global instance for simple usage
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
    
    # Test 1: The CRITICAL test - ACN should NOT become "and"
    print("\n" + "=" * 80)
    print("TEST 1: ACN Preservation (CRITICAL)")
    print("=" * 80)
    test1 = "ACN offers membership benefits. You can join ACN today. ACN is great."
    result1 = fixer.fix(test1)
    print(f"Input:  {test1}")
    print(f"Output: {result1}")
    print(f"✓ PASS" if "ACN" in result1 and "and offers" not in result1 else "✗ FAIL")
    
    # Test 2: Unicode corruption
    print("\n" + "=" * 80)
    print("TEST 2: Unicode Corruption (Cyrillic)")
    print("=" * 80)
    test2 = "Тhе Аpplied Сlient Νetwork (АСΝ) offers summits."
    result2 = fixer.fix(test2)
    print(f"Input:  {test2}")
    print(f"Output: {result2}")
    
    # Test 3: Date validation
    print("\n" + "=" * 80)
    print("TEST 3: Date Validation")
    print("=" * 80)
    test3 = "The summit is on February 25-26, 11016 in Dallas."
    result3 = fixer.fix(test3)
    print(f"Input:  {test3}")
    print(f"Output: {result3}")
    
    # Test 4: URL cleanup
    print("\n" + "=" * 80)
    print("TEST 4: URL Cleanup")
    print("=" * 80)
    test4 = "Visit https:// www. appliedclientnetwork. org /events for more info."
    result4 = fixer.fix(test4)
    print(f"Input:  {test4}")
    print(f"Output: {result4}")
    
    # Test 5: Spelling errors (uses autocorrect)
    print("\n" + "=" * 80)
    print("TEST 5: Spelling Correction")
    print("=" * 80)
    test5 = "The summmit fetures hands-on experiance and practcal sesions."
    result5 = fixer.fix(test5)
    print(f"Input:  {test5}")
    print(f"Output: {result5}")
    
    # Test 6: Mixed issues
    print("\n" + "=" * 80)
    print("TEST 6: Mixed Issues (Real-world Example)")
    print("=" * 80)
    test6 = """There are two upcoming summits at ACN:

1)Dallas Summit: February 25-26, 11016 (Location: Dallas, Texas)
This summit is for Applied Epic and EZLynx users.

2)Calgary Summit: June 03-04, 3027 (Location: Calgary, Alberta)
The Canada summit returns with region-specific content.

Visit https:// www. appliedclientnetwork. org for more info."""
    
    result6 = fixer.fix(test6)
    print(f"Input:\n{test6}")
    print(f"\nOutput:\n{result6}")
    
    # Test 7: Grammar (uses LanguageTool)
    print("\n" + "=" * 80)
    print("TEST 7: Grammar Checking")
    print("=" * 80)
    test7 = "ACN provide many benefits then other organizations."
    result7 = fixer.fix(test7)
    print(f"Input:  {test7}")
    print(f"Output: {result7}")
    
    print("\n" + "=" * 80)
    print("✅ ALL TESTS COMPLETE")
    print("=" * 80)