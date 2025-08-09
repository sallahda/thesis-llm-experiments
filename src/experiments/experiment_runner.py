import os
import sys
import logging
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ExperimentRunner:
    """Orchestrates all LLM experiments for vulnerability narrative generation."""
    
    def __init__(self, results_dir: str = "./results"):
        self.results_dir = Path(results_dir)
        self.setup_directories()
        
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is required for judge LLM")
    
    def setup_directories(self) -> None:
        """Create results directories for each experiment."""
        directories = [
            self.results_dir / "clarity",
            self.results_dir / "accuracy", 
            self.results_dir / "stakeholder-specificity"
        ]
        
        for dir_path in directories:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Results directories created in: {self.results_dir}")
    
    def run_clarity_experiment(self) -> Dict[str, Any]:
        """Run the clarity/readability experiment."""
        logger.info("Starting Clarity Experiment...")
        
        try:
            from clarity_experiment import ClarityExperiment
            
            experiment = ClarityExperiment()
            results_df = experiment.run_experiments()
            
            results_df.to_csv(self.results_dir / "clarity" / "clarity_results.csv", index=False)
            
            analysis = experiment.analyze_results(results_df)
            with open(self.results_dir / "clarity" / "clarity_analysis.json", 'w') as f:
                json.dump(analysis, f, indent=2, default=str)
            
            logger.info(f"Clarity experiment completed: {len(results_df)} narratives evaluated")
            
            return {
                'success': True,
                'narratives_count': len(results_df),
                'mean_fk_score': analysis['overall']['mean_fk_score'],
                'target_met_pct': analysis['overall']['target_met_percentage']
            }
            
        except Exception as e:
            logger.error(f"Clarity experiment failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def run_accuracy_experiment(self) -> Dict[str, Any]:
        """Run the accuracy experiment using Judge LLM."""
        logger.info("Starting Accuracy Experiment...")
        
        try:
            from accuracy_experiment import AccuracyExperiment
            
            experiment = AccuracyExperiment()
            
            experiment.generate_all_narratives()
            logger.info(f"Generated {len(experiment.narratives)} narratives")
            
            results_df = experiment.evaluate_accuracy()
            
            results_df.to_csv(self.results_dir / "accuracy" / "accuracy_results.csv", index=False)
            
            analysis = experiment.analyze_results(results_df)
            with open(self.results_dir / "accuracy" / "accuracy_analysis.json", 'w') as f:
                json.dump(analysis, f, indent=2, default=str)
            
            valid_evaluations = len(results_df[results_df['evaluation_successful'] == True])
            logger.info(f"Accuracy experiment completed: {valid_evaluations} valid evaluations")
            
            return {
                'success': True,
                'total_evaluations': len(results_df),
                'valid_evaluations': valid_evaluations,
                'mean_accuracy': analysis['overall']['mean_accuracy'] if 'overall' in analysis else 0,
                'high_accuracy_pct': analysis['overall']['high_accuracy_percentage'] if 'overall' in analysis else 0
            }
            
        except Exception as e:
            logger.error(f"Accuracy experiment failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def run_stakeholder_experiment(self) -> Dict[str, Any]:
        """Run the stakeholder specificity experiment."""
        logger.info("Starting Stakeholder Specificity Experiment...")
        
        try:
            from stakeholder_experiment import StakeholderExperiment
            
            experiment = StakeholderExperiment()
            
            experiment.generate_all_narratives()
            logger.info(f"Generated {len(experiment.narratives)} narratives")
            
            results_df = experiment.evaluate_stakeholder_specificity()
            
            results_df.to_csv(self.results_dir / "stakeholder-specificity" / "stakeholder_results.csv", index=False)
            
            analysis = experiment.analyze_results(results_df)
            with open(self.results_dir / "stakeholder-specificity" / "stakeholder_analysis.json", 'w') as f:
                json.dump(analysis, f, indent=2, default=str)
            
            valid_evaluations = len(results_df[results_df['evaluation_successful'] == True])
            logger.info(f"Stakeholder experiment completed: {valid_evaluations} valid evaluations")
            
            return {
                'success': True,
                'total_evaluations': len(results_df),
                'valid_evaluations': valid_evaluations,
                'mean_stakeholder_score': analysis['overall']['mean_stakeholder_score'] if 'overall' in analysis else 0,
                'high_alignment_pct': analysis['overall']['high_alignment_percentage'] if 'overall' in analysis else 0
            }
            
        except Exception as e:
            logger.error(f"Stakeholder experiment failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def create_experiment_summary(self, clarity_result: Dict, accuracy_result: Dict, 
                                stakeholder_result: Dict) -> Dict:
        """Create overall experiment summary."""
        
        summary = {
            'experiment_run': {
                'timestamp': datetime.now().isoformat(),
                'experiments_completed': sum([
                    clarity_result.get('success', False),
                    accuracy_result.get('success', False), 
                    stakeholder_result.get('success', False)
                ])
            },
            'clarity_experiment': clarity_result,
            'accuracy_experiment': accuracy_result,
            'stakeholder_experiment': stakeholder_result,
            'next_steps': [
                "Run results_analysis.py for comprehensive analysis",
                "Check individual experiment files for detailed results",
                "Review generated visualizations and metrics"
            ]
        }
        
        with open(self.results_dir / "experiment_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        return summary
    
    def run_all_experiments(self) -> Dict:
        """Run all three experiments in sequence."""
        logger.info("Starting Complete LLM Experiments Pipeline")
        logger.info("="*70)
        logger.info("Experiments: Clarity → Accuracy → Stakeholder Specificity")
        logger.info("="*70)
        
        start_time = datetime.now()
        
        clarity_result = self.run_clarity_experiment()
        accuracy_result = self.run_accuracy_experiment()
        stakeholder_result = self.run_stakeholder_experiment()
        
        summary = self.create_experiment_summary(clarity_result, accuracy_result, stakeholder_result)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        # Print final summary
        print("\n" + "="*70)
        print("EXPERIMENT PIPELINE SUMMARY")
        print("="*70)
        
        print(f"Total Duration: {duration}")
        print(f"Experiments Completed: {summary['experiment_run']['experiments_completed']}/3")
        
        if clarity_result['success']:
            print(f"Clarity: {clarity_result['narratives_count']} narratives, "
                  f"{clarity_result['target_met_pct']:.1f}% target achievement")
        else:
            print(f"Clarity: FAILED - {clarity_result.get('error', 'Unknown error')}")
        
        if accuracy_result['success']:
            print(f"Accuracy: {accuracy_result['valid_evaluations']} valid evaluations, "
                  f"{accuracy_result['high_accuracy_pct']:.1f}% high accuracy")
        else:
            print(f"Accuracy: FAILED - {accuracy_result.get('error', 'Unknown error')}")
        
        if stakeholder_result['success']:
            print(f"Stakeholder: {stakeholder_result['valid_evaluations']} valid evaluations, "
                  f"{stakeholder_result['high_alignment_pct']:.1f}% high alignment")
        else:
            print(f"Stakeholder: FAILED - {stakeholder_result.get('error', 'Unknown error')}")
        
        print(f"\nResults saved to: {self.results_dir}")
        print(f"Summary file: experiment_summary.json")
        
        if summary['experiment_run']['experiments_completed'] >= 2:
            print(f"\nReady for analysis!")
            print(f"Next step: Run results_analysis.py for comprehensive visualization and insights")
        else:
            print(f"\nSome experiments failed. Check logs and retry failed experiments.")
        
        print("="*70)
        
        return summary

def main():
    """Main execution function."""
    try:
        if not os.getenv("OPENAI_API_KEY"):
            print("Error: OPENAI_API_KEY environment variable is required")
            print("Set it with: export OPENAI_API_KEY='your-api-key-here'")
            sys.exit(1)
        
        runner = ExperimentRunner()
        summary = runner.run_all_experiments()
        
        return summary
        
    except KeyboardInterrupt:
        print("\nExperiments interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
