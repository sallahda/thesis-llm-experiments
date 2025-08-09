import boto3
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import time
import logging
from datetime import datetime
import numpy as np
import os

from judge_llm import JudgeLLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AccuracyExperiment:
    """Evaluates factual accuracy of LLM-generated vulnerability narratives using GPT-4 as judge."""
    
    def __init__(self, region_name: str = "eu-west-1", openai_api_key: Optional[str] = None):
        self.region_name = region_name
        self.bedrock = boto3.client("bedrock-runtime", region_name=region_name)
        self.judge = JudgeLLM(model_name="gpt-4")
        
        self.models = {
            "claude_sonnet": "eu.anthropic.claude-3-7-sonnet-20250219-v1:0",
            "nova_pro": "eu.amazon.nova-pro-v1:0", 
            "llama_3b": "eu.meta.llama3-2-3b-instruct-v1:0"
        }
        
        self.visualizations = {
            "stacked_bar": {
                "type": "Stacked Bar Chart",
                "description": "Shows vulnerability distribution by severity across AWS accounts",
                "sample_data": {
                    "accounts": ["account1", "account2", "account3"],
                    "critical": [5, 12, 3],
                    "high": [8, 15, 7],
                    "medium": [20, 25, 18],
                    "low": [35, 40, 22]
                }
            },
            "heatmap": {
                "type": "Labeled Heatmap", 
                "description": "Displays vulnerability severity scores with CVE labels across ECS clusters",
                "sample_data": {
                    "clusters": ["web-cluster", "api-cluster", "db-cluster"],
                    "vulnerabilities": ["CVE-2024-1234", "CVE-2024-5678", "CVE-2024-9012"],
                    "scores": [[9.8, 7.5, 0], [8.2, 9.1, 6.8], [0, 5.4, 8.9]]
                }
            },
            "treemap": {
                "type": "Treemap",
                "description": "Hierarchical view of vulnerability distribution across services and packages",
                "sample_data": {
                    "hierarchy": {
                        "web-service": {
                            "nginx": {"CVE-2024-1111": 8.5, "CVE-2024-2222": 6.2},
                            "openssl": {"CVE-2024-3333": 9.1}
                        },
                        "api-service": {
                            "python": {"CVE-2024-4444": 7.8},
                            "flask": {"CVE-2024-5555": 5.9}
                        }
                    }
                }
            }
        }
        
        self.narratives: List[Dict] = []
        self.results: List[Dict] = []
        
    def get_prompt_strategies(self) -> Dict[str, str]:
        """The four prompt engineering strategies used in the thesis experiments."""
        return {
            "general": """Explain this vulnerability visualization clearly and concisely.""",
            
            "role_based": """You are a security analyst reporting to management. 
            Explain this vulnerability visualization in a way that helps decision-makers 
            understand the security posture and required actions.""",
            
            "few_shot": """Here are two examples of good vulnerability explanations:

            Example 1: "This bar chart shows 47 critical vulnerabilities across three AWS accounts. 
            Account2 has the highest risk with 12 critical issues requiring immediate patching."
            
            Example 2: "The heatmap reveals CVE-2024-1234 affects multiple clusters with scores above 8.0. 
            Priority should be given to patching the web-cluster and api-cluster first."
            
            Now explain this vulnerability visualization:""",
            
            "chain_of_thought": """Please analyze this vulnerability visualization step by step:
            1. First, identify the key elements and data points shown
            2. Then, explain what these elements represent in terms of security risk
            3. Finally, summarize the main insight and recommended actions
            
            Explain this vulnerability visualization:"""
        }

    def generate_narrative(self, model_id: str, prompt: str, visualization_data: Dict) -> str:
        """Generate narrative using specified LLM and prompt."""
        viz_context = f"""
        Visualization Type: {visualization_data['type']}
        Description: {visualization_data['description']}
        Data: {json.dumps(visualization_data['sample_data'], indent=2)}
        """
        
        full_prompt = f"{prompt}\n\n{viz_context}"
        
        try:
            response = self.bedrock.converse(
                modelId=model_id,
                messages=[{"role": "user", "content": [{"text": full_prompt}]}],
                inferenceConfig={"temperature": 0.1, "maxTokens": 500}
            )
            return response["output"]["message"]["content"][0]["text"]
        except Exception as e:
            logger.error(f"Error generating narrative with {model_id}: {str(e)}")
            return f"Error generating narrative: {str(e)}"

    def generate_all_narratives(self) -> List[Dict]:
        """Generate all 36 narratives (3 models × 4 prompts × 3 visualizations)."""
        logger.info("Generating narratives for accuracy evaluation...")
        
        prompt_strategies = self.get_prompt_strategies()
        narrative_count = 0
        total_narratives = len(self.models) * len(prompt_strategies) * len(self.visualizations)
        narratives = []
        
        for model_name, model_id in self.models.items():
            for prompt_name, prompt_text in prompt_strategies.items():
                for viz_name, viz_data in self.visualizations.items():
                    narrative_count += 1
                    logger.info(f"Generating narrative {narrative_count}/{total_narratives}: "
                              f"{model_name} + {prompt_name} + {viz_name}")
                    
                    start_time = time.time()
                    narrative = self.generate_narrative(model_id, prompt_text, viz_data)
                    generation_time = time.time() - start_time
                    
                    narrative_data = {
                        'narrative_id': narrative_count,
                        'model': model_name,
                        'prompt_strategy': prompt_name,
                        'visualization': viz_name,
                        'narrative': narrative,
                        'visualization_data': viz_data,
                        'generation_time': generation_time,
                        'timestamp': datetime.now().isoformat()
                    }
                    narratives.append(narrative_data)
                    time.sleep(0.5)
        
        self.narratives = narratives
        logger.info(f"Generated all {total_narratives} narratives")
        return narratives

    def evaluate_accuracy(self) -> pd.DataFrame:
        """Evaluate narratives for factual accuracy using Judge LLM."""
        logger.info("Evaluating narratives for factual accuracy...")
        
        if not self.narratives:
            raise ValueError("No narratives generated. Run generate_all_narratives() first.")
        
        evaluation_configs = []
        for narrative_data in self.narratives:
            config = {
                'narrative': narrative_data['narrative'],
                'visualization_data': narrative_data['visualization_data'],
                'context': f"This is a {narrative_data['visualization_data']['type']} visualization"
            }
            evaluation_configs.append(config)
        
        accuracy_results = self.judge.batch_evaluate(
            evaluation_configs, 
            evaluation_type="accuracy",
            delay=1.0
        )
        
        for i, result in enumerate(accuracy_results):
            narrative_data = self.narratives[i]
            combined_result = {
                **narrative_data,
                'accuracy_score': result['score'],
                'accuracy_reasoning': result['reasoning'],
                'judge_tokens_used': result['tokens_used'],
                'evaluation_successful': result['score'] is not None
            }
            self.results.append(combined_result)
        
        df_results = pd.DataFrame(self.results)
        logger.info(f"Completed accuracy evaluation for all {len(self.results)} narratives")
        return df_results

    def analyze_results(self, df: pd.DataFrame) -> Dict:
        """Perform statistical analysis of accuracy results."""
        logger.info("Analyzing accuracy results...")
        
        df_valid = df[df['evaluation_successful'] == True].copy()
        
        if len(df_valid) == 0:
            logger.error("No valid evaluations to analyze")
            return {}
        
        analysis = {
            'overall': {
                'mean_accuracy': df_valid['accuracy_score'].mean(),
                'std_accuracy': df_valid['accuracy_score'].std(),
                'high_accuracy_percentage': (df_valid['accuracy_score'] >= 4).sum() / len(df_valid) * 100,
                'best_score': df_valid['accuracy_score'].max(),
                'worst_score': df_valid['accuracy_score'].min(),
                'total_valid_evaluations': len(df_valid),
                'total_failed_evaluations': len(df) - len(df_valid)
            },
            'by_model': df_valid.groupby('model')['accuracy_score'].agg(['mean', 'std', 'count']).round(2),
            'by_prompt': df_valid.groupby('prompt_strategy')['accuracy_score'].agg(['mean', 'std', 'count']).round(2),
            'by_visualization': df_valid.groupby('visualization')['accuracy_score'].agg(['mean', 'std', 'count']).round(2),
            'best_combinations': df_valid.nlargest(5, 'accuracy_score')[
                ['model', 'prompt_strategy', 'visualization', 'accuracy_score', 'accuracy_reasoning']
            ]
        }
        
        try:
            model_groups = [group['accuracy_score'].values for name, group in df_valid.groupby('model')]
            f_stat_model, p_val_model = stats.f_oneway(*model_groups)
            
            prompt_groups = [group['accuracy_score'].values for name, group in df_valid.groupby('prompt_strategy')]
            f_stat_prompt, p_val_prompt = stats.f_oneway(*prompt_groups)
            
            analysis['anova'] = {
                'model_f_stat': f_stat_model,
                'model_p_value': p_val_model,
                'model_significant': p_val_model < 0.05,
                'prompt_f_stat': f_stat_prompt,
                'prompt_p_value': p_val_prompt,
                'prompt_significant': p_val_prompt < 0.05
            }
        except Exception as e:
            logger.warning(f"ANOVA test failed: {e}")
            analysis['anova'] = None
        
        return analysis

    def create_visualizations(self, df: pd.DataFrame, analysis: Dict) -> None:
        """Create accuracy analysis visualizations."""
        logger.info("Creating accuracy visualizations...")
        
        df_valid = df[df['evaluation_successful'] == True].copy()
        
        if len(df_valid) == 0:
            logger.error("No valid results to visualize")
            return
        
        plt.style.use('default')
        sns.set_palette("husl")
        
        fig = plt.figure(figsize=(20, 16))
        
        # Distribution of accuracy scores
        ax1 = plt.subplot(3, 3, 1)
        plt.hist(df_valid['accuracy_score'], bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], 
                alpha=0.7, color='lightblue', edgecolor='black')
        plt.axvspan(4, 5, alpha=0.3, color='green', label='High Accuracy (4-5)')
        plt.xlabel('Accuracy Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of Accuracy Scores')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Accuracy by model
        ax2 = plt.subplot(3, 3, 2)
        df_valid.boxplot(column='accuracy_score', by='model', ax=ax2)
        plt.axhspan(4, 5, alpha=0.3, color='green')
        plt.title('Accuracy Scores by LLM Model')
        plt.xlabel('Model')
        plt.ylabel('Accuracy Score')
        plt.xticks(rotation=45)
        
        # Continue with other visualizations...
        # (keeping the rest of the visualization code but cleaning up comments)
        
        plt.tight_layout()
        plt.savefig('./results/accuracy/accuracy_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("Visualizations saved as 'accuracy_results.png'")

    def save_results(self, df: pd.DataFrame, analysis: Dict) -> None:
        """Save results to CSV and JSON files."""
        logger.info("Saving accuracy results...")
        
        df.to_csv('./results/accuracy/accuracy_results.csv', index=False)
        
        with open('./results/accuracy/accuracy_analysis.json', 'w') as f:
            json_analysis = {}
            for key, value in analysis.items():
                if key == 'anova' and value is not None:
                    json_analysis[key] = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                                        for k, v in value.items()}
                elif isinstance(value, dict):
                    json_analysis[key] = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                                        for k, v in value.items()}
                else:
                    json_analysis[key] = value
            
            json.dump(json_analysis, f, indent=2, default=str)
        
        logger.info("Results saved to CSV and JSON files")

    def run_experiment(self) -> pd.DataFrame:
        """Run the complete accuracy experiment pipeline."""
        logger.info("Starting LLM Accuracy Experiment")
        print("=" * 60)
        
        try:
            self.generate_all_narratives()
            results_df = self.evaluate_accuracy()
            analysis = self.analyze_results(results_df)
            
            print("\nACCURACY EXPERIMENT SUMMARY")
            print("=" * 60)
            print(f"Total evaluations: {len(results_df)}")
            print(f"Valid evaluations: {analysis['overall']['total_valid_evaluations']}")
            print(f"Mean accuracy: {analysis['overall']['mean_accuracy']:.2f}")
            print(f"High accuracy rate: {analysis['overall']['high_accuracy_percentage']:.1f}%")
            print(f"Best score: {analysis['overall']['best_score']:.0f}")
            print(f"Worst score: {analysis['overall']['worst_score']:.0f}")
            
            if analysis['anova']:
                print(f"\nStatistical significance:")
                print(f"Model differences: {'Yes' if analysis['anova']['model_significant'] else 'No'}")
                print(f"Prompt differences: {'Yes' if analysis['anova']['prompt_significant'] else 'No'}")
            
            self.create_visualizations(results_df, analysis)
            self.save_results(results_df, analysis)
            
            print("\nAccuracy experiment completed successfully!")
            print("Check the generated files:")
            print("   - accuracy_results.png (visualization)")
            print("   - accuracy_results.csv (raw data)")
            print("   - accuracy_analysis.json (summary stats)")
            
            return results_df
            
        except Exception as e:
            logger.error(f"Accuracy experiment failed: {str(e)}")
            raise

def main():
    """Run the accuracy experiment."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set your OPENAI_API_KEY environment variable")
        return
    
    experiment = AccuracyExperiment(openai_api_key=api_key)
    results = experiment.run_experiment()
    return results

if __name__ == "__main__":
    main()
