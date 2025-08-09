import boto3
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import textstat
from typing import Dict, List
import time
import logging
from datetime import datetime
import numpy as np
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClarityExperiment:
    """Evaluates readability of LLM-generated vulnerability narratives using Flesch-Kincaid scoring."""
    
    def __init__(self, region_name: str = "eu-west-1"):
        self.region_name = region_name
        self.bedrock = boto3.client("bedrock-runtime", region_name=region_name)
        
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

    def calculate_readability_score(self, text: str) -> float:
        """Calculate Flesch-Kincaid grade level for readability assessment."""
        return textstat.flesch_kincaid_grade(text)

    def run_experiments(self) -> pd.DataFrame:
        """Run all 36 combinations of experiments (3 models × 4 prompts × 3 visualizations)."""
        logger.info("Starting LLM clarity experiments...")
        
        prompt_strategies = self.get_prompt_strategies()
        experiment_count = 0
        total_experiments = len(self.models) * len(prompt_strategies) * len(self.visualizations)
        
        for model_name, model_id in self.models.items():
            for prompt_name, prompt_text in prompt_strategies.items():
                for viz_name, viz_data in self.visualizations.items():
                    experiment_count += 1
                    logger.info(f"Experiment {experiment_count}/{total_experiments}: "
                              f"{model_name} + {prompt_name} + {viz_name}")
                    
                    start_time = time.time()
                    narrative = self.generate_narrative(model_id, prompt_text, viz_data)
                    generation_time = time.time() - start_time
                    
                    fk_score = self.calculate_readability_score(narrative)
                    
                    result = {
                        'experiment_id': experiment_count,
                        'model': model_name,
                        'prompt_strategy': prompt_name,
                        'visualization': viz_name,
                        'narrative': narrative,
                        'fk_score': fk_score,
                        'generation_time': generation_time,
                        'target_met': 9 <= fk_score <= 15,  # Target range for readability
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    self.results.append(result)
                    time.sleep(0.5)
        
        df_results = pd.DataFrame(self.results)
        logger.info(f"Completed all {total_experiments} experiments")
        return df_results

    def analyze_results(self, df: pd.DataFrame) -> Dict:
        """Perform statistical analysis of readability results."""
        logger.info("Analyzing results...")
        
        analysis = {
            'overall': {
                'mean_fk_score': df['fk_score'].mean(),
                'std_fk_score': df['fk_score'].std(),
                'target_met_percentage': (df['target_met'].sum() / len(df)) * 100,
                'best_score': df['fk_score'].min(),  # Lower FK score = better readability
                'worst_score': df['fk_score'].max()
            },
            'by_model': df.groupby('model')['fk_score'].agg(['mean', 'std', 'count']).round(2),
            'by_prompt': df.groupby('prompt_strategy')['fk_score'].agg(['mean', 'std', 'count']).round(2),
            'by_visualization': df.groupby('visualization')['fk_score'].agg(['mean', 'std', 'count']).round(2)
        }
        
        # Find best combinations (closest to target range 9-12)
        df['distance_from_target'] = df['fk_score'].apply(
            lambda x: 0 if 9 <= x <= 12 else min(abs(x - 9), abs(x - 12))
        )
        
        analysis['best_combinations'] = df.nsmallest(5, 'distance_from_target')[
            ['model', 'prompt_strategy', 'visualization', 'fk_score', 'target_met']
        ]
        
        # Statistical significance testing
        try:
            model_groups = [group['fk_score'].values for name, group in df.groupby('model')]
            f_stat_model, p_val_model = stats.f_oneway(*model_groups)
            
            prompt_groups = [group['fk_score'].values for name, group in df.groupby('prompt_strategy')]
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
        """Create clarity analysis visualizations."""
        logger.info("Creating visualizations...")
        
        plt.style.use('default')
        sns.set_palette("husl")
        
        fig = plt.figure(figsize=(20, 16))
        
        # FK Score Distribution
        ax1 = plt.subplot(3, 3, 1)
        plt.hist(df['fk_score'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvspan(9, 12, alpha=0.3, color='green', label='Target Range (9-12)')
        plt.xlabel('Flesch-Kincaid Grade Level')
        plt.ylabel('Frequency')
        plt.title('Distribution of FK Scores Across All Experiments')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # FK Scores by Model
        ax2 = plt.subplot(3, 3, 2)
        df.boxplot(column='fk_score', by='model', ax=ax2)
        plt.axhspan(9, 12, alpha=0.3, color='green')
        plt.title('FK Scores by LLM Model')
        plt.xlabel('Model')
        plt.ylabel('Flesch-Kincaid Grade Level')
        plt.xticks(rotation=45)
        
        # Continue with other visualizations as needed...
        
        plt.tight_layout()
        plt.savefig('./results/clarity/clarity_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("Visualizations saved as 'clarity_results.png'")

    def save_results(self, df: pd.DataFrame, analysis: Dict) -> None:
        """Save results to CSV and JSON files."""
        logger.info("Saving results...")
        
        df.to_csv('./results/clarity/clarity_results.csv', index=False)
        
        with open('./results/clarity/clarity_analysis.json', 'w') as f:
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

def main():
    """Run the clarity experiment."""
    print("Starting LLM Clarity Experiments for Vulnerability Narratives")
    print("=" * 60)
    
    experiment = ClarityExperiment()
    
    try:
        results_df = experiment.run_experiments()
        analysis = experiment.analyze_results(results_df)
        
        print("\nEXPERIMENT SUMMARY")
        print("=" * 60)
        print(f"Total experiments: {len(results_df)}")
        print(f"Mean FK Score: {analysis['overall']['mean_fk_score']:.2f}")
        print(f"Target met: {analysis['overall']['target_met_percentage']:.1f}%")
        print(f"Best score: {analysis['overall']['best_score']:.2f}")
        print(f"Worst score: {analysis['overall']['worst_score']:.2f}")
        
        if analysis['anova']:
            print(f"\nStatistical significance:")
            print(f"Model differences: {'Yes' if analysis['anova']['model_significant'] else 'No'}")
            print(f"Prompt differences: {'Yes' if analysis['anova']['prompt_significant'] else 'No'}")
        
        experiment.create_visualizations(results_df, analysis)
        experiment.save_results(results_df, analysis)
        
        print("\nExperiments completed successfully!")
        print("Check the generated files:")
        print("   - clarity_results.png (visualization)")
        print("   - clarity_results.csv (raw data)")
        print("   - clarity_analysis.json (summary stats)")
        
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
