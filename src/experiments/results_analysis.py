import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import sys
import logging
from typing import Dict, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMResultsAnalyzer:
    """Analyze LLM experiment results with metric-focused visualizations."""
    
    def __init__(self, results_dir: str = "./results"):
        self.results_dir = Path(results_dir)
        self.models = ["claude_sonnet", "nova_pro", "llama_3b"] 
        self.prompts = ["general", "role_based", "few_shot", "chain_of_thought"]
        self.stakeholder_groups = ["Product Owners", "Development Teams", "Security Specialists"]
        
        self.validate_results_exist()
    
    def validate_results_exist(self) -> None:
        """Check that experiment results exist."""
        required_files = [
            self.results_dir / "clarity" / "clarity_results.csv",
            self.results_dir / "accuracy" / "accuracy_results.csv",
            self.results_dir / "stakeholder-specificity" / "stakeholder_results.csv"
        ]
        
        missing_files = [f for f in required_files if not f.exists()]
        
        if missing_files:
            print("Missing experiment results files:")
            for file in missing_files:
                print(f"   {file}")
            print("\nRun 'python experiment_runner.py' first to generate results")
            sys.exit(1)
        
        logger.info("All experiment result files found")
    
    def load_experiment_results(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load results from all three experiments."""
        logger.info("Loading experiment results...")
        
        clarity_df = pd.read_csv(self.results_dir / "clarity" / "clarity_results.csv")
        logger.info(f"   Clarity: {len(clarity_df)} narratives")
        
        accuracy_df = pd.read_csv(self.results_dir / "accuracy" / "accuracy_results.csv")
        accuracy_df = accuracy_df[accuracy_df['evaluation_successful'] == True]
        logger.info(f"   Accuracy: {len(accuracy_df)} valid evaluations")
        
        stakeholder_df = pd.read_csv(self.results_dir / "stakeholder-specificity" / "stakeholder_results.csv")
        stakeholder_df = stakeholder_df[stakeholder_df['evaluation_successful'] == True]
        logger.info(f"   Stakeholder: {len(stakeholder_df)} valid evaluations")
        
        return clarity_df, accuracy_df, stakeholder_df
    
    def prepare_stakeholder_data(self, stakeholder_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Aggregate stakeholder results by narrative for cross-stakeholder analysis."""
        logger.info("Preparing stakeholder data...")
        
        # Average scores across all stakeholders for each narrative
        stakeholder_by_narrative = stakeholder_df.groupby([
            'narrative_id', 'model', 'prompt_strategy', 'visualization'
        ]).agg({
            'stakeholder_score': 'mean',
            'evaluation_successful': 'all'
        }).reset_index()
        
        # Create individual stakeholder breakdowns
        stakeholder_breakdown = {}
        for stakeholder in self.stakeholder_groups:
            stakeholder_data = stakeholder_df[stakeholder_df['stakeholder_group'] == stakeholder]
            stakeholder_breakdown[stakeholder] = stakeholder_data
        
        logger.info(f"   Aggregated to {len(stakeholder_by_narrative)} narrative-level scores")
        
        return stakeholder_by_narrative, stakeholder_breakdown
    
    def create_metric_focused_charts(self, clarity_df: pd.DataFrame, accuracy_df: pd.DataFrame, 
                                   stakeholder_by_narrative: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create the main charts per metric visualization."""
        logger.info("Creating metric-focused charts...")
        
        plt.style.use('default')
        fig, axes = plt.subplots(1, 3, figsize=(22, 7))
        
        model_colors = {
            'claude_sonnet': '#2E86AB',
            'nova_pro': '#A23B72',
            'llama_3b': '#F18F01'
        }
        
        model_order = ['claude_sonnet', 'nova_pro', 'llama_3b']
        
        # 1. CLARITY PERFORMANCE (FK Scores - Lower is better for readability)
        ax1 = axes[0]
        clarity_summary = clarity_df.groupby(['prompt_strategy', 'model'])['fk_score'].mean().unstack()
        clarity_summary = clarity_summary.reindex(columns=model_order)
        
        bars1 = clarity_summary.plot(kind='bar', ax=ax1, 
                                     color=[model_colors[col] for col in clarity_summary.columns],
                                     width=0.8, alpha=0.8)
        
        ax1.axhspan(9, 15, alpha=0.2, color='green', label='Target Range (9-15)', zorder=0)
        ax1.set_title('Clarity Performance by Prompt Strategy\n(Target Range: 9-15 FK Grade Level)', 
                    fontsize=14, fontweight='bold')
        ax1.set_ylabel('Flesch-Kincaid Grade Level', fontsize=12)
        ax1.set_xlabel('Prompt Strategy', fontsize=12)
        ax1.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim(0, max(clarity_df['fk_score'].max() * 1.1, 15))
        
        for container in ax1.containers:
            ax1.bar_label(container, fmt='%.1f', fontsize=9)
        
        # 2. ACCURACY PERFORMANCE (1-5 scale - Higher is better)
        ax2 = axes[1]
        accuracy_summary = accuracy_df.groupby(['prompt_strategy', 'model'])['accuracy_score'].mean().unstack()
        accuracy_summary = accuracy_summary.reindex(columns=model_order)
        
        bars2 = accuracy_summary.plot(kind='bar', ax=ax2,
                                     color=[model_colors[col] for col in accuracy_summary.columns],
                                     width=0.8, alpha=0.8)
        
        ax2.axhspan(4, 5, alpha=0.2, color='green', label='High Accuracy (≥4)', zorder=0)
        ax2.set_title('Accuracy Performance by Prompt Strategy\n(Higher Score = More Accurate)', 
                     fontsize=14, fontweight='bold')
        ax2.set_ylabel('Accuracy Score (1-5)', fontsize=12)
        ax2.set_xlabel('Prompt Strategy', fontsize=12)
        ax2.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_ylim(1, 5.3)
        ax2.grid(True, alpha=0.3, axis='y')
        
        for container in ax2.containers:
            ax2.bar_label(container, fmt='%.2f', fontsize=9)
        
        # 3. STAKEHOLDER ALIGNMENT (1-5 scale - Higher is better)
        ax3 = axes[2]
        stakeholder_summary = stakeholder_by_narrative.groupby(['prompt_strategy', 'model'])['stakeholder_score'].mean().unstack()
        stakeholder_summary = stakeholder_summary.reindex(columns=model_order)
        
        bars3 = stakeholder_summary.plot(kind='bar', ax=ax3,
                                        color=[model_colors[col] for col in stakeholder_summary.columns],
                                        width=0.8, alpha=0.8)
        
        ax3.axhspan(4, 5, alpha=0.2, color='green', label='High Alignment (≥4)', zorder=0)
        ax3.set_title('Stakeholder Alignment by Prompt Strategy\n(Higher Score = Better Fit)', 
                     fontsize=14, fontweight='bold')
        ax3.set_ylabel('Stakeholder Fit Score (1-5)', fontsize=12)
        ax3.set_xlabel('Prompt Strategy', fontsize=12)
        ax3.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.tick_params(axis='x', rotation=45)
        ax3.set_ylim(1, 5)
        ax3.grid(True, alpha=0.3, axis='y')
        
        for container in ax3.containers:
            ax3.bar_label(container, fmt='%.2f', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'llm_metric_focused_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("   Saved: llm_metric_focused_analysis.png")
        
        return clarity_summary, accuracy_summary, stakeholder_summary
    
    def create_cross_metric_analysis(self, clarity_df: pd.DataFrame, accuracy_df: pd.DataFrame, 
                                   stakeholder_by_narrative: pd.DataFrame) -> pd.DataFrame:
        """Create analysis showing which combinations work across multiple metrics."""
        logger.info("Creating cross-metric analysis...")
        
        cross_analysis = []
        
        for model in self.models:
            for prompt in self.prompts:
                clarity_data = clarity_df[(clarity_df['model'] == model) & (clarity_df['prompt_strategy'] == prompt)]
                accuracy_data = accuracy_df[(accuracy_df['model'] == model) & (accuracy_df['prompt_strategy'] == prompt)]
                stakeholder_data = stakeholder_by_narrative[(stakeholder_by_narrative['model'] == model) & (stakeholder_by_narrative['prompt_strategy'] == prompt)]
                
                if len(clarity_data) > 0 and len(accuracy_data) > 0 and len(stakeholder_data) > 0:
                    avg_fk = clarity_data['fk_score'].mean()
                    avg_accuracy = accuracy_data['accuracy_score'].mean()
                    avg_stakeholder = stakeholder_data['stakeholder_score'].mean()
                    
                    clarity_target_met = (clarity_data['fk_score'].between(9, 15)).mean() * 100
                    accuracy_target_met = (accuracy_data['accuracy_score'] >= 4).mean() * 100
                    stakeholder_target_met = (stakeholder_data['stakeholder_score'] >= 4).mean() * 100
                    
                    readability_score = max(0, 15 - abs(avg_fk - 12))
                    
                    cross_analysis.append({
                        'model': model,
                        'prompt_strategy': prompt,
                        'combination': f"{model}_{prompt}",
                        'avg_fk_score': avg_fk,
                        'avg_accuracy': avg_accuracy,
                        'avg_stakeholder': avg_stakeholder,
                        'readability_score': readability_score,
                        'clarity_target_pct': clarity_target_met,
                        'accuracy_target_pct': accuracy_target_met,
                        'stakeholder_target_pct': stakeholder_target_met,
                        'overall_target_pct': (clarity_target_met + accuracy_target_met + stakeholder_target_met) / 3
                    })
        
        cross_df = pd.DataFrame(cross_analysis)
        
        # Create cross-metric visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Target Achievement Rates Heatmap
        ax1 = axes[0, 0]
        heatmap_data = cross_df.pivot(index='model', columns='prompt_strategy', values='overall_target_pct')
        sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax1, 
                   vmin=0, vmax=100, cbar_kws={'label': 'Achievement Rate (%)'})
        ax1.set_title('Overall Target Achievement Rate\n(Average Across All Metrics)', fontweight='bold')
        
        # 2. Best Performers Bar Chart
        ax2 = axes[0, 1]
        top_combinations = cross_df.nlargest(8, 'overall_target_pct')
        bars = ax2.barh(range(len(top_combinations)), top_combinations['overall_target_pct'], 
                       color='lightgreen', alpha=0.8)
        ax2.set_yticks(range(len(top_combinations)))
        ax2.set_yticklabels([f"{row['model'][:6]}\n{row['prompt_strategy'][:8]}" 
                            for _, row in top_combinations.iterrows()])
        ax2.set_xlabel('Overall Achievement Rate (%)')
        ax2.set_title('Top Performing Combinations\n(Cross-Metric)', fontweight='bold')
        ax2.axvline(x=60, color='orange', linestyle='--', alpha=0.7, label='Good Performance')
        ax2.axvline(x=80, color='green', linestyle='--', alpha=0.7, label='Excellent Performance')
        ax2.legend()
        
        for i, (bar, row) in enumerate(zip(bars, top_combinations.itertuples())):
            ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                    f'{row.overall_target_pct:.1f}%', va='center', fontsize=9)
        
        # 3. Trade-offs Scatter Plot
        ax3 = axes[1, 0]
        colors = [{'claude_sonnet': 'blue', 'nova_pro': 'red', 'llama_3b': 'green'}[model] 
                 for model in cross_df['model']]
        scatter = ax3.scatter(cross_df['avg_accuracy'], cross_df['avg_stakeholder'], 
                             c=colors, alpha=0.7, s=100)
        ax3.axhline(y=4, color='green', linestyle='--', alpha=0.5)
        ax3.axvline(x=4, color='green', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Average Accuracy Score')
        ax3.set_ylabel('Average Stakeholder Score') 
        ax3.set_title('Accuracy vs Stakeholder Alignment\nTrade-offs', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        for _, row in cross_df.iterrows():
            if row['avg_accuracy'] >= 4 and row['avg_stakeholder'] >= 4:
                ax3.annotate(f"{row['model'][:4]}\n{row['prompt_strategy'][:4]}", 
                           (row['avg_accuracy'], row['avg_stakeholder']),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 4. Clarity vs Performance Trade-off
        ax4 = axes[1, 1]
        colors = [{'claude_sonnet': 'blue', 'nova_pro': 'red', 'llama_3b': 'green'}[model] 
                 for model in cross_df['model']]
        scatter2 = ax4.scatter(cross_df['readability_score'], cross_df['avg_accuracy'], 
                              c=colors, alpha=0.7, s=100)
        ax4.set_xlabel('Readability Score (Higher = More Readable)')
        ax4.set_ylabel('Average Accuracy Score')
        ax4.set_title('Readability vs Accuracy\nTrade-offs', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='blue', label='Claude Sonnet'),
                          Patch(facecolor='red', label='Nova Pro'),
                          Patch(facecolor='green', label='Llama 3B')]
        ax4.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'llm_cross_metric_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("   Saved: llm_cross_metric_analysis.png")
        
        return cross_df
    
    def generate_optimization_insights(self, cross_df: pd.DataFrame) -> Dict:
        """Generate insights for answering the research sub-question."""
        logger.info("Generating optimization insights...")
        
        insights = {
            'best_clarity': {'combination': None, 'fk_score': float('inf'), 'details': {}},
            'best_accuracy': {'combination': None, 'score': 0, 'details': {}},
            'best_stakeholder': {'combination': None, 'score': 0, 'details': {}}
        }
        
        for _, row in cross_df.iterrows():
            combo = f"{row['model']} + {row['prompt_strategy']}"
            
            # Best clarity (FK score closest to 12.0)
            fk_distance = abs(row['avg_fk_score'] - 12.0)
            if fk_distance < abs(insights['best_clarity']['fk_score'] - 12.0):
                insights['best_clarity'] = {
                    'combination': combo,
                    'fk_score': row['avg_fk_score'],
                    'details': row
                }
            
            # Best accuracy
            if row['avg_accuracy'] > insights['best_accuracy']['score']:
                insights['best_accuracy'] = {
                    'combination': combo,
                    'score': row['avg_accuracy'],
                    'details': row
                }
            
            # Best stakeholder alignment
            if row['avg_stakeholder'] > insights['best_stakeholder']['score']:
                insights['best_stakeholder'] = {
                    'combination': combo,
                    'score': row['avg_stakeholder'],
                    'details': row
                }
        
        # Best overall (balanced across all metrics)
        best_overall = cross_df.loc[cross_df['overall_target_pct'].idxmax()]
        insights['best_overall'] = {
            'combination': f"{best_overall['model']} + {best_overall['prompt_strategy']}",
            'achievement_rate': best_overall['overall_target_pct'],
            'details': best_overall
        }
        
        # High performers
        high_performers = cross_df[
            (cross_df['clarity_target_pct'] >= 50) & 
            (cross_df['accuracy_target_pct'] >= 60) & 
            (cross_df['stakeholder_target_pct'] >= 60)
        ].sort_values('overall_target_pct', ascending=False)
        
        insights['high_performers'] = high_performers.head(5)
        
        return insights
    
    def create_thesis_summary(self, insights: Dict, cross_df: pd.DataFrame) -> Dict:
        """Create final summary for thesis."""
        logger.info("Creating thesis summary...")
        
        thesis_summary = {
            'research_question': "How can LLMs + prompt engineering optimally generate clear, accurate, and stakeholder-specific narratives?",
            'key_findings': {
                'best_for_clarity': {
                    'combination': insights['best_clarity']['combination'],
                    'fk_score': round(insights['best_clarity']['fk_score'], 1),
                    'interpretation': 'Closest to ideal 9-15 grade reading level'
                },
                'best_for_accuracy': {
                    'combination': insights['best_accuracy']['combination'],
                    'score': round(insights['best_accuracy']['score'], 2),
                    'interpretation': 'Highest factual accuracy rating'
                },
                'best_for_stakeholders': {
                    'combination': insights['best_stakeholder']['combination'],
                    'score': round(insights['best_stakeholder']['score'], 2),
                    'interpretation': 'Best alignment with stakeholder needs'
                },
                'best_overall': {
                    'combination': insights['best_overall']['combination'],
                    'achievement_rate': round(insights['best_overall']['achievement_rate'], 1),
                    'interpretation': 'Best balanced performance across all criteria'
                }
            },
            'optimization_insights': {
                'prompt_strategy_impact': cross_df.groupby('prompt_strategy')['overall_target_pct'].mean().to_dict(),
                'model_impact': cross_df.groupby('model')['overall_target_pct'].mean().to_dict(),
                'high_performers_count': len(insights['high_performers'])
            },
            'recommendations': {
                'for_clarity_priority': insights['best_clarity']['combination'],
                'for_accuracy_priority': insights['best_accuracy']['combination'], 
                'for_stakeholder_priority': insights['best_stakeholder']['combination'],
                'for_balanced_performance': insights['best_overall']['combination']
            }
        }
        
        with open(self.results_dir / 'thesis_optimization_analysis.json', 'w') as f:
            json.dump(thesis_summary, f, indent=2, default=str)
        
        cross_df.to_csv(self.results_dir / 'llm_cross_metric_performance.csv', index=False)
        if len(insights['high_performers']) > 0:
            insights['high_performers'].to_csv(self.results_dir / 'llm_high_performers.csv', index=False)
        
        logger.info("   Saved: thesis_optimization_analysis.json")
        logger.info("   Saved: llm_cross_metric_performance.csv")
        
        return thesis_summary
        
    def run_complete_analysis(self) -> Tuple[pd.DataFrame, Dict, Dict]:
        """Run the complete analysis pipeline."""
        print("Starting LLM Results Analysis")
        print("="*60)
        print("Research Focus: Optimal LLM + Prompt Engineering for Vulnerability Narratives")
        print("="*60)
        
        try:
            clarity_df, accuracy_df, stakeholder_df = self.load_experiment_results()
            
            stakeholder_by_narrative, stakeholder_breakdown = self.prepare_stakeholder_data(stakeholder_df)
            
            clarity_summary, accuracy_summary, stakeholder_summary = self.create_metric_focused_charts(
                clarity_df, accuracy_df, stakeholder_by_narrative
            )
            
            cross_df = self.create_cross_metric_analysis(clarity_df, accuracy_df, stakeholder_by_narrative)
            
            insights = self.generate_optimization_insights(cross_df)
            
            thesis_summary = self.create_thesis_summary(insights, cross_df)
            
            print("\n" + "="*60)
            print("KEY OPTIMIZATION FINDINGS")
            print("="*60)
            
            print(f"BEST PERFORMERS BY CRITERION:")
            print(f"   Clarity: {insights['best_clarity']['combination']} (FK: {insights['best_clarity']['fk_score']:.1f})")
            print(f"   Accuracy: {insights['best_accuracy']['combination']} (Score: {insights['best_accuracy']['score']:.2f}/5)")
            print(f"   Stakeholder Fit: {insights['best_stakeholder']['combination']} (Score: {insights['best_stakeholder']['score']:.2f}/5)")
            print(f"   Overall Balance: {insights['best_overall']['combination']} (Achievement: {insights['best_overall']['achievement_rate']:.1f}%)")
            
            print(f"\nHIGH PERFORMERS (Multi-Criteria):")
            if len(insights['high_performers']) > 0:
                for i, (_, row) in enumerate(insights['high_performers'].head(3).iterrows()):
                    print(f"   {i+1}. {row['model']} + {row['prompt_strategy']} (Overall: {row['overall_target_pct']:.1f}%)")
            else:
                print("   No combinations met high performance criteria across all metrics")
            
            print(f"\nOPTIMIZATION INSIGHTS:")
            prompt_performance = cross_df.groupby('prompt_strategy')['overall_target_pct'].mean()
            model_performance = cross_df.groupby('model')['overall_target_pct'].mean()
            
            best_prompt = prompt_performance.idxmax()
            best_model = model_performance.idxmax()
            
            print(f"   Best Prompt Strategy: {best_prompt} ({prompt_performance[best_prompt]:.1f}% avg achievement)")
            print(f"   Best Model: {best_model} ({model_performance[best_model]:.1f}% avg achievement)")
            
            print(f"\nFILES CREATED FOR THESIS:")
            print(f"   llm_metric_focused_analysis.png (main visualization)")
            print(f"   llm_cross_metric_analysis.png (trade-offs & optimization)")
            print(f"   llm_cross_metric_performance.csv (detailed results)")
            if len(insights['high_performers']) > 0:
                print(f"   llm_high_performers.csv (top combinations)")
            print(f"   thesis_optimization_analysis.json (summary)")
            
            print("\nAnalysis completed successfully!")
            print("Results ready for thesis Sub-Question 3 discussion")
            print("="*60)
            
            return cross_df, insights, thesis_summary
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            raise

def main():
    """Main execution function."""
    try:
        analyzer = LLMResultsAnalyzer()
        cross_df, insights, thesis_summary = analyzer.run_complete_analysis()
        return cross_df, insights, thesis_summary
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Analysis failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
