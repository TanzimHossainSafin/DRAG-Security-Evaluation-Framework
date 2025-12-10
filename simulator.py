import json
import random
import sys
from typing import List

from datasets import load_dataset
from jsonargparse import Namespace
from loguru import logger
import numpy as np
from tqdm import tqdm

from modules.exp_logger import ExpLogger
from modules.data_types import Datapoint, Testcase
from modules.rag_network import DRAGNetwork, CRAGNetwork, NoRAGNetwork
from modules.evaluator import QAEvaluator
from modules.options import parse_args

# new
from modules.attacks import DataPoisoningAttack, MembershipInferenceAttack


def get_nested_value(data_dict: dict, dot_key_path: str):
    """
    Retrieves a nested value from a dictionary using a dot-separated key path.

    Args:
        data_dict: The dictionary to retrieve the value from.
        dot_key_path: A string representing the nested keys separated by dots (e.g., "key1.key2.key3").

    Returns:
        The value at the specified path in the dictionary.
    """
    keys = dot_key_path.split(".")
    value = data_dict
    for key in keys:
        value = value[key]
    return value


def run_simulation(cfg: Namespace):
    # Init csv logger
    exp_logger = ExpLogger()
    logger.info(f"Experiment Log Directory: {exp_logger.experiment_dir}")
    config_logger = exp_logger.get_yaml_logger("config")
    metrics_logger = exp_logger.get_csv_logger("metrics")
    test_cases_logger = exp_logger.get_csv_logger("test_cases")

    # Save all config
    config_logger.log(cfg.as_dict())
    config_logger.save()

    # Load Huggingface dataset
    dataset = load_dataset(**cfg.data.load.as_dict())
    data_points: List[Datapoint] = []
    all_topics = set()

    # task type:
    # - mcqa for Multiple Choice Question Answering
    # - ogqa for Open Generative Question Answering
    task_type = cfg.data.task_type

    if cfg.rag.test_mode:
        # Only pick 20 samples from dataset in test mode
        dataset = dataset.select(range(20))
    else:
        if cfg.data.num_samples is not None:
            # Sample data if num_samples is specified
            dataset = dataset.shuffle(seed=cfg.rag.random_seed).take(min(cfg.data.num_samples, len(dataset)))
        else:
            dataset = dataset.shuffle(seed=cfg.rag.random_seed)

    # Prepare data points
    for item in dataset:
        topic = get_nested_value(item, cfg.data.topic_path)
        question = get_nested_value(item, cfg.data.question_path)
        answer = get_nested_value(item, cfg.data.answer_path)
        if task_type == "mcqa":
            choices = get_nested_value(item, cfg.data.choices_path)
            connection_term = " Select the best answer from the following candidates, replying with 1, 2, 3, or 4: "
            question = str(question) + connection_term + str(choices)

        data_point = Datapoint(topic=str(topic), question=str(question), answer=str(answer))
        all_topics.add(str(topic))
        data_points.append(data_point)
    
    if cfg.rag.network_type == "DRAG":
        filtered_data_points = data_points
    elif cfg.rag.network_type == "CRAG":
        # Filter out a portion of data points in CRAG for comparison
        num_topics_to_keep = int(len(all_topics) * (1.0 - cfg.rag.filter_out_topic_ratio))
        filtered_topics = random.sample(list(all_topics), k=num_topics_to_keep)
        filtered_data_points = [
            dp for dp in data_points if dp.topic in filtered_topics
        ]
        num_datapoints_to_keep = int(len(filtered_data_points) * (1.0 - cfg.rag.filter_out_qa_ratio))
        filtered_data_points = random.sample(filtered_data_points, k=num_datapoints_to_keep)
    elif cfg.rag.network_type == "NoRAG":
        filtered_data_points = []
    else:
        raise ValueError(f"Unknown network type: {cfg.rag.network_type}")

    # Initialize DRAG parameters
    query_confidence_threshold = cfg.rag.query_confidence_threshold
    num_query_neighbor = min(cfg.rag.num_query_neighbor, cfg.rag.num_peers - 1)
    query_ttl = cfg.rag.query_ttl

    # Initialize RAG network with peers and knowledges
    if cfg.rag.network_type == "DRAG":
        rag_net = DRAGNetwork(cfg.rag.num_peers, cfg.rag.num_peer_attachments, cfg.llm.base_url, cfg.llm.name, 
                              cfg.llm.num_ctx, cfg.rag.random_seed)
    elif cfg.rag.network_type == "CRAG":
        rag_net = CRAGNetwork(cfg.llm.base_url, cfg.llm.name, cfg.llm.num_ctx, cfg.rag.random_seed)
    elif cfg.rag.network_type == "NoRAG":
        rag_net = NoRAGNetwork(cfg.llm.base_url, cfg.llm.name, cfg.llm.num_ctx, cfg.rag.random_seed)
    else:
        raise ValueError(f"Unknown network type: {cfg.rag.network_type}")
    rag_net.init_knowledge(filtered_data_points)

    
    # === ATTACK SIMULATION ===
    attack_results = None
    mia_results = None
    
    # Data Poisoning Attack
    if cfg.security.enable_attack:
        logger.info("=" * 50)
        logger.info("DATA POISONING ATTACK ENABLED")
        logger.info("=" * 50)
        
        # Create attack instance
        attack = DataPoisoningAttack(
            poisoning_ratio=cfg.security.poisoning_ratio,
            attack_strategy=cfg.security.attack_strategy,
            poison_type=cfg.security.poison_type,
            target_peer_ids=cfg.security.get('target_peer_ids', []),  # For targeted attack
            amplification_factor=cfg.security.get('amplification_factor', 3),  # NEW
            question_variants=cfg.security.get('question_variants', 2)         # NEW
        )
        
        # Execute attack
        attack_results = attack.execute(rag_net, filtered_data_points)
        
        # Log attack details
        attack_logger = exp_logger.get_yaml_logger("attack_config")
        attack_logger.log(attack_results)
        attack_logger.save()
        
        logger.info(f"Attack executed: {attack_results['num_poisoned_peers']} peers poisoned")
    
    # Membership Inference Attack
    if cfg.security.get('enable_membership_inference', False):
        logger.info("=" * 50)
        logger.info("MEMBERSHIP INFERENCE ATTACK ENABLED")
        logger.info("=" * 50)
        
        # Create MIA instance
        mia_attack = MembershipInferenceAttack(
            inference_method=cfg.security.get('mia_inference_method', 'confidence_based'),
            test_size=cfg.security.get('mia_test_size', 0.5),
            threshold_percentile=cfg.security.get('mia_threshold_percentile', 50),
            random_seed=cfg.security.get('mia_random_seed', 42)
        )
        
        # Execute MIA
        mia_results = mia_attack.execute(rag_net, data_points)
        
        # Log MIA details
        mia_logger = exp_logger.get_yaml_logger("mia_config")
        mia_logger.log(mia_results)
        mia_logger.save()
        
        logger.info(f"MIA executed: Accuracy={mia_results['attack_accuracy']:.2%}, "
                   f"Privacy Risk={mia_results['privacy_risk']}")


    # Run evaluation
    qa_evaluator = QAEvaluator()

    for idx, data_point in enumerate(tqdm(data_points, desc=f"Inferencing on {len(data_points)} test case(s)")):
        if cfg.rag.network_type == "DRAG":
            if cfg.rag.search_algorithm == "TARW":
                rag_answer = rag_net.topic_query(
                    data_point.question, 
                    num_query_neighbor=num_query_neighbor, 
                    query_confidence_threshold=query_confidence_threshold,
                    max_ttl=query_ttl
                )
            elif cfg.rag.search_algorithm == "RW":
                rag_answer = rag_net.random_walk_query(
                    data_point.question,
                    query_confidence_threshold=query_confidence_threshold,
                    max_ttl=query_ttl
                )
            elif cfg.rag.search_algorithm == "FL":
                rag_answer = rag_net.flooding_query(
                    data_point.question,
                    query_confidence_threshold=query_confidence_threshold,
                    max_ttl=query_ttl
                )
            else:
                raise ValueError(f"Unkonw search algorithm: {cfg.rag.search_algorithm}")
        elif cfg.rag.network_type == "CRAG":
            rag_answer = rag_net.query(
                data_point.question,
                query_confidence_threshold=query_confidence_threshold
            )
        elif cfg.rag.network_type == "NoRAG":
            rag_answer = rag_net.query(
                data_point.question
            )
        else:
            raise ValueError(f"Unknown network type: {cfg.rag.network_type}")

        # test case
        test_case = Testcase(
            question=data_point.question,
            expected_output=data_point.answer,
            actual_output=rag_answer.answer,
            relevant_knowledge=rag_answer.relevant_knowledge,
            relevant_score=rag_answer.relevant_score,
            num_hops=rag_answer.num_hops,
            num_messages=rag_answer.num_messages,
            is_query_hit=rag_answer.is_query_hit
        )
        test_cases_logger.log(test_case.model_dump())
        qa_evaluator.add(test_case)

        # log evaluation results regularly
        if idx % cfg.rag.log_every_n_steps == 0:
            test_cases_logger.save()
            eval_results = qa_evaluator.get_results()
            metrics_logger.log(eval_results)
            metrics_logger.save()

    # log final results
    test_cases_logger.save()
    eval_results = qa_evaluator.get_results()
    metrics_logger.log(eval_results)
    metrics_logger.save()

    # Evaluate attack success if attack was performed
    if attack_results is not None:
        logger.info("=" * 50)
        logger.info("EVALUATING DATA POISONING ATTACK SUCCESS")
        logger.info("=" * 50)
        
        # For comparison, you would need baseline metrics from a clean run
        # For now, we'll just log the attacked metrics
        attack_eval = {
            "attack_config": attack_results,
            "attacked_metrics": eval_results,
            "performance_impact": {
                "f1_score": eval_results.get("f1", 0.0),
                "exact_match": eval_results.get("exact_match", 0.0),
                "semantic_similarity": eval_results.get("semantic_similarity", 0.0)
            }
        }
        
        attack_eval_logger = exp_logger.get_yaml_logger("attack_evaluation")
        attack_eval_logger.log(attack_eval)
        attack_eval_logger.save()
        
        logger.info(f"\nAttack Evaluation:\n{json.dumps(attack_eval, indent=2)}\n")
    
    # Evaluate MIA success if MIA was performed
    if mia_results is not None:
        logger.info("=" * 50)
        logger.info("EVALUATING MEMBERSHIP INFERENCE ATTACK SUCCESS")
        logger.info("=" * 50)
        
        mia_eval = {
            "attack_results": mia_results,
            "interpretation": {
                "attack_success": mia_results['attack_accuracy'] > 0.5,
                "privacy_leak_detected": mia_results['attack_accuracy'] > 0.6,
                "attack_accuracy": f"{mia_results['attack_accuracy']:.2%}",
                "privacy_risk": mia_results['privacy_risk']
            }
        }
        
        mia_eval_logger = exp_logger.get_yaml_logger("mia_evaluation")
        mia_eval_logger.log(mia_eval)
        mia_eval_logger.save()
        
        logger.info(f"\nMIA Evaluation:\n{json.dumps(mia_eval, indent=2)}\n")

    logger.info(f"\nFinal Evaluation Results:\n{json.dumps(eval_results)}\n")


def main():
    # parse arguments
    cfg = parse_args()

    # Initialize random seeds
    random.seed(cfg.rag.random_seed)
    np.random.seed(cfg.rag.random_seed)

    # Changing the level of the logger
    logger.remove()  # Remove default handler.
    logger.add(sys.stderr, level=cfg.log_level)

    # run evaluation
    run_simulation(cfg)


if __name__ == "__main__":
    main()

