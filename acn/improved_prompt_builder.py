#!/usr/bin/env python3
"""
Enhanced Prompt Builder with Chain-of-Thought and ReAct patterns
Features:
- Chain-of-Thought (CoT): Encourages step-by-step reasoning
- ReAct: Reason + Act framework for better understanding
- Intelligent context extraction
- Temporal awareness
- Fact-grounding validation
"""

from dataclasses import dataclass
from typing import Optional
from datetime import datetime


@dataclass
class QueryIntent:
    """Structured query intent"""
    category: str  # membership, events, howto, resources, general
    is_temporal: bool
    temporal_mode: Optional[str]  # upcoming, past, specific_year, all
    year: Optional[int] = None
    confidence: float = 0.0


class ImprovedPromptBuilder:
    """Build intelligent prompts with reasoning chains"""
    
    @staticmethod
    def build(
        question: str,
        context: str,
        intent: QueryIntent,
        current_date: str
    ) -> str:
        """Build optimized prompt with chain-of-thought and ReAct"""
        
        # Step 1: Build thinking instructions (Chain-of-Thought)
        thinking_instructions = ImprovedPromptBuilder._build_thinking_instructions(intent)
        
        # Step 2: Build ReAct framework
        react_framework = ImprovedPromptBuilder._build_react_framework(intent, current_date)
        
        # Step 3: Build category-specific instructions
        specific_instructions = ImprovedPromptBuilder._build_specific_instructions(intent, current_date)
        
        # Step 4: Build validation rules
        validation_rules = ImprovedPromptBuilder._build_validation_rules(intent)
        
        # Step 5: Assemble complete prompt
        prompt = f"""You are a helpful assistant for Applied Client Network (ACN).
Today's date: {current_date}

{thinking_instructions}

{react_framework}

{specific_instructions}

{validation_rules}

<context>
{context}
</context>

Question: {question}

Let's work through this step by step:

1. THINK: Analyze what the question is asking for
2. REASON: What information from the context is relevant?
3. ACT: Formulate the answer based only on context
4. VALIDATE: Ensure the answer is factually grounded

Answer:"""
        
        return prompt
    
    @staticmethod
    def _build_thinking_instructions(intent: QueryIntent) -> str:
        """Build chain-of-thought thinking instructions"""
        
        thinking = """THINKING FRAMEWORK - Think step-by-step:
Step 1: What is the user specifically asking?
Step 2: What category of information do they need?
Step 3: What key facts must be extracted from the context?
Step 4: Are there dates/times that matter for this query?
Step 5: Is any important information missing from the context?"""
        
        if intent.category == "events":
            thinking += """
Step 6: What are the event dates? Are they BEFORE or AFTER today?
Step 7: Which events match the temporal requirement (upcoming/past/specific year)?"""
        
        elif intent.category == "membership":
            thinking += """
Step 6: What are the key membership benefits?
Step 7: What is the cost/pricing structure?"""
        
        elif intent.category == "howto":
            thinking += """
Step 6: What are the numbered steps?
Step 7: Are there prerequisites mentioned?
Step 8: Are there direct resource links?"""
        
        return thinking
    
    @staticmethod
    def _build_react_framework(intent: QueryIntent, current_date: str) -> str:
        """Build ReAct (Reason + Act) framework"""
        
        react = f"""REACT FRAMEWORK - Reason and Act:

REASON:
- What does the question require?
- What context is available?
- Are there temporal constraints (upcoming/past)?
- What facts are EXPLICITLY stated vs. inferred?

ACT:
- Extract only facts explicitly stated in context
- Do NOT invent details
- Do NOT infer beyond what's written
- For temporal queries: Compare dates to TODAY ({current_date})
- Validate each claim against the source material"""
        
        return react
    
    @staticmethod
    def _build_specific_instructions(intent: QueryIntent, current_date: str) -> str:
        """Build category-specific instructions"""
        
        if intent.category == "membership":
            instructions = """
MEMBERSHIP QUERY INSTRUCTIONS:
- List membership benefits clearly and specifically
- Include pricing/cost information if available
- Explain how to join step-by-step
- Mention renewal process if relevant
- Use bullet points for benefits
- DO NOT add information not in context
- DO NOT mention generic benefits not explicitly stated"""
        
        elif intent.category == "events":
            instructions = f"""
EVENT QUERY INSTRUCTIONS:
- CRITICAL: Today is {current_date}
- Extract event name, date, time, location for each event
- For "UPCOMING" queries: ONLY include events with dates >= {current_date}
- For "PAST" queries: ONLY include events with dates < {current_date}
- For "SPECIFIC YEAR" queries: ONLY include events from that year
- List each event with: Name | Date | Time | Location | Details
- Include registration/signup links if available
- If dates are ambiguous or missing, state "Date not specified"
- DO NOT assume future events if date is unclear
- DO NOT include events outside the requested timeframe"""
        
        elif intent.category == "howto":
            instructions = """
HOW-TO QUERY INSTRUCTIONS:
- Provide numbered, step-by-step instructions
- Include prerequisites or requirements
- Link to relevant resources mentioned in context
- Explain any technical terms
- DO NOT add steps not in the context
- Use clear, sequential numbering"""
        
        else:
            instructions = """
GENERAL QUERY INSTRUCTIONS:
- Provide accurate, factual information
- Preserve original wording and structure
- Be specific and detailed
- DO NOT paraphrase aggressively
- DO NOT merge or alter technical terms
- DO NOT add opinions or editorializing"""
        
        return instructions
    
    @staticmethod
    def _build_validation_rules(intent: QueryIntent) -> str:
        """Build fact validation rules"""
        
        rules = """
VALIDATION RULES - BEFORE ANSWERING:
✓ MUST: Answer based ONLY on the provided context
✓ MUST: Include specific dates/times when discussing events
✓ MUST: State "I don't have that information" for missing details
✓ MUST: Ground every claim in context with a reference
✓ MUST: Preserve exact names, dates, and technical terminology

✗ MUST NOT: Invent facts, names, or details not in context
✗ MUST NOT: Merge words or alter terminology
✗ MUST NOT: Add opinions, editorializing, or interpretation
✗ MUST NOT: Mention people/organizations not explicitly named
✗ MUST NOT: Assume dates or fill in missing information
✗ MUST NOT: Hallucinate links or resources not mentioned

TEMPORAL VALIDATION (for events):
- Extract the exact date from context
- Compare to today's date
- If temporal qualifier doesn't match (e.g., "upcoming" but event is past), EXCLUDE it
- If date is unclear, state "Date information not available"

FACT VALIDATION:
- Every claim must be explicitly present in context
- Paraphrase is acceptable, but not addition
- If context is incomplete, acknowledge: "Based on available information..."
- When details are sparse, provide what IS available"""
        
        return rules
    
    @staticmethod
    def build_with_evidence_tracking(
        question: str,
        context: str,
        intent: QueryIntent,
        current_date: str
    ) -> tuple:
        """Build prompt with evidence tracking for fact-grounding
        
        Returns:
            (prompt, evidence_template) - prompt and a template for tracking evidence
        """
        
        prompt = ImprovedPromptBuilder.build(question, context, intent, current_date)
        
        # Add evidence tracking section
        evidence_tracking = """
For each claim in your answer, cite the source:
Format: [CLAIM]: [EVIDENCE FROM CONTEXT]

Example:
- Dallas Summit is on March 15, 2026: [Evidence: "Dallas Summit scheduled for March 15, 2026"]
- Registration closes February 28: [Evidence: "Registration deadline: February 28"]"""
        
        prompt += f"\n\n{evidence_tracking}"
        
        return prompt, evidence_tracking
    
    @staticmethod
    def build_for_strict_grounding(
        question: str,
        context: str,
        intent: QueryIntent,
        current_date: str
    ) -> str:
        """Build ultra-strict fact-grounded prompt"""
        
        prompt = f"""You are a fact-grounded assistant for Applied Client Network (ACN).
Today's date: {current_date}

STRICT GROUNDING MODE - Answer with evidence references:

Your ONLY job is to extract and relay information from the context below.
Do NOT generate, invent, or infer anything.

For each fact you mention, you MUST indicate its source from the context.
Format: "Claim (Source: excerpt from context)"

PROHIBITED:
- Inferring dates or times not explicitly stated
- Adding details not in the provided context
- Making assumptions about past or future
- Mentioning names not explicitly in the context
- Paraphrasing aggressively or merging information

REQUIRED:
- Cite context for every claim
- Use exact dates and names from context
- State "This information is not available in the context" for gaps
- Acknowledge if information is incomplete

<context>
{context}
</context>

Question: {question}

Answer with evidence references:"""
        
        return prompt


class PromptBuilderRegistry:
    """Registry of prompt builders for different strategies"""
    
    BUILDERS = {
        "standard": ImprovedPromptBuilder.build,
        "evidence_tracked": ImprovedPromptBuilder.build_with_evidence_tracking,
        "strict": ImprovedPromptBuilder.build_for_strict_grounding,
    }
    
    @classmethod
    def build(cls, strategy: str = "standard", **kwargs) -> str:
        """Build prompt using specified strategy"""
        
        if strategy == "evidence_tracked":
            prompt, _ = cls.BUILDERS[strategy](**kwargs)
            return prompt
        else:
            builder = cls.BUILDERS.get(strategy, ImprovedPromptBuilder.build)
            return builder(**kwargs)


# ==================== TEST ====================

if __name__ == "__main__":
    
    # Test data
    sample_context = """
    The Dallas Summit 2026 is scheduled for March 15-17, 2026, at the Dallas Convention Center.
    This annual event brings together ACN members to discuss industry trends and best practices.
    Registration opens February 1 and closes March 1, 2026.
    Early bird pricing: $499 (available until February 15)
    Regular pricing: $699
    """
    
    question = "When is the Dallas Summit?"
    
    intent = QueryIntent(
        category="events",
        is_temporal=True,
        temporal_mode="upcoming",
        confidence=0.95
    )
    
    current_date = "January 31, 2026"
    
    print("="*80)
    print("IMPROVED PROMPT WITH CHAIN-OF-THOUGHT AND REACT")
    print("="*80)
    
    prompt = ImprovedPromptBuilder.build(question, context=sample_context, intent=intent, current_date=current_date)
    
    print(prompt)
    
    print("\n" + "="*80)
    print("STRICT GROUNDING PROMPT")
    print("="*80)
    
    strict_prompt = ImprovedPromptBuilder.build_for_strict_grounding(question, context=sample_context, intent=intent, current_date=current_date)
    
    print(strict_prompt)
