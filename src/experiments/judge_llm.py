from openai import OpenAI
import json
import logging
import time
from typing import Dict, List, Optional
import re
import os

logger = logging.getLogger(__name__)

class JudgeLLM:
    """LLM judge for evaluating narrative quality across different dimensions."""
    
    def __init__(self, model_name: str = "gpt-4", api_key: Optional[str] = None):
        self.model_name = model_name
        
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = OpenAI()
        
        try:
            self._test_connection()
            logger.info(f"Judge LLM initialized successfully with {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Judge LLM: {e}")
            raise
    
    def _test_connection(self) -> str:
        """Test the connection to the LLM API."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=5,
            temperature=0
        )
        return response.choices[0].message.content
    
    def evaluate_accuracy(self, narrative: str, visualization_data: Dict, 
                         context: str = "") -> Dict:
        """
        Evaluate the factual accuracy of a narrative.
        
        Returns dict with score (1-5), reasoning, and metadata.
        """
        
        prompt = f"""You are an expert cybersecurity analyst evaluating the factual accuracy of vulnerability report narratives.

                    TASK: Rate how accurately this narrative interprets the vulnerability visualization data.

                    VISUALIZATION DATA:
                    {json.dumps(visualization_data, indent=2)}

                    CONTEXT: {context}

                    NARRATIVE TO EVALUATE:
                    "{narrative}"

                    EVALUATION CRITERIA:
                    - Does the narrative correctly interpret the data values?
                    - Are the vulnerability counts, severities, and trends accurate?
                    - Does it make any false claims about the data?
                    - Are technical terms used correctly?

                    SCORING SCALE:
                    1 = Major factual errors, misinterprets key data points
                    2 = Some factual errors, mostly correct but notable mistakes  
                    3 = Mostly accurate, minor errors or imprecisions
                    4 = Accurate interpretation, very minor issues if any
                    5 = Completely accurate, perfect interpretation of the data

                    Respond in this exact format:
                    SCORE: [1-5]
                    REASONING: [Brief explanation of your evaluation]
                """

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.1
            )
            
            response_text = response.choices[0].message.content
            score, reasoning = self._parse_response(response_text)
            
            return {
                'score': score,
                'reasoning': reasoning,
                'response_text': response_text,
                'tokens_used': response.usage.total_tokens if hasattr(response, 'usage') else None
            }
            
        except Exception as e:
            logger.error(f"Error evaluating accuracy: {e}")
            return {
                'score': None,
                'reasoning': f"Evaluation failed: {str(e)}",
                'response_text': None,
                'tokens_used': None
            }
    
    def evaluate_stakeholder_specificity(self, narrative: str, stakeholder_group: str,
                                       visualization_data: Dict, context: str = "") -> Dict:
        """
        Evaluate how well a narrative addresses specific stakeholder needs.
        
        Returns dict with score (1-5), reasoning, and metadata.
        """
        
        stakeholder_needs = {
            "Product Owners": "business impact, prioritization guidance, resource allocation decisions",
            "Development Teams": "actionable remediation steps, technical details, implementation guidance", 
            "Security Specialists": "risk assessment, comprehensive analysis, strategic security insights"
        }
        
        needs = stakeholder_needs.get(stakeholder_group, "general cybersecurity insights")
        
        prompt = f"""You are an expert in cybersecurity communication evaluating how well a vulnerability narrative addresses specific stakeholder needs.

                    TARGET STAKEHOLDER: {stakeholder_group}
                    STAKEHOLDER NEEDS: {needs}

                    VISUALIZATION DATA:
                    {json.dumps(visualization_data, indent=2)}

                    CONTEXT: {context}

                    NARRATIVE TO EVALUATE:
                    "{narrative}"

                    EVALUATION CRITERIA:
                    - Does the narrative address the specific needs of {stakeholder_group}?
                    - Is the technical level appropriate for this audience?
                    - Does it provide the type of insights this stakeholder would find actionable?
                    - Is the language and focus suitable for their decision-making role?

                    SCORING SCALE:
                    1 = Completely misaligned with stakeholder needs
                    2 = Partially relevant but misses key stakeholder concerns
                    3 = Generally appropriate, some alignment with stakeholder needs
                    4 = Well-aligned, addresses most stakeholder requirements
                    5 = Perfectly tailored, directly addresses all key stakeholder needs

                    Respond in this exact format:
                    SCORE: [1-5]
                    REASONING: [Brief explanation of your evaluation]
                """

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.1
            )
            
            response_text = response.choices[0].message.content
            score, reasoning = self._parse_response(response_text)
            
            return {
                'score': score,
                'reasoning': reasoning,
                'response_text': response_text,
                'stakeholder_group': stakeholder_group,
                'tokens_used': response.usage.total_tokens if hasattr(response, 'usage') else None
            }
            
        except Exception as e:
            logger.error(f"Error evaluating stakeholder specificity: {e}")
            return {
                'score': None,
                'reasoning': f"Evaluation failed: {str(e)}",
                'response_text': None,
                'stakeholder_group': stakeholder_group,
                'tokens_used': None
            }
    
    def _parse_response(self, response_text: str) -> tuple:
        """Parse the structured response from the judge LLM."""
        try:
            score_match = re.search(r'SCORE:\s*(\d)', response_text)
            score = int(score_match.group(1)) if score_match else None
            
            reasoning_match = re.search(r'REASONING:\s*(.+?)(?=\n\n|\Z)', response_text, re.DOTALL)
            reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"
            
            return score, reasoning
            
        except Exception as e:
            logger.error(f"Error parsing judge response: {e}")
            return None, f"Parse error: {str(e)}"
    
    def batch_evaluate(self, evaluations: List[Dict], evaluation_type: str = "accuracy",
                      delay: float = 1.0) -> List[Dict]:
        """
        Perform batch evaluation with rate limiting.
        
        Args:
            evaluations: List of evaluation configs
            evaluation_type: 'accuracy' or 'stakeholder_specificity'
            delay: Delay between API calls in seconds
            
        Returns:
            List of evaluation results
        """
        results = []
        
        for i, eval_config in enumerate(evaluations):
            logger.info(f"Evaluating {i+1}/{len(evaluations)}: {evaluation_type}")
            
            if evaluation_type == "accuracy":
                result = self.evaluate_accuracy(**eval_config)
            elif evaluation_type == "stakeholder_specificity":
                result = self.evaluate_stakeholder_specificity(**eval_config)
            else:
                raise ValueError(f"Unknown evaluation type: {evaluation_type}")
            
            result['evaluation_id'] = i + 1
            results.append(result)
            
            if i < len(evaluations) - 1:
                time.sleep(delay)
        
        return results
