from jsonargparse import ArgumentParser

def parse_args():
    parser = ArgumentParser(
        default_config_files=[
            "./config/rag.yaml", 
            "./config/llm/llama32_3b.yaml", 
            "./config/data/mmlu.yaml",
            "./config/security.yaml"  # New
        ]
    )

    parser.add_argument('--log_level', type=str, help='DEBUG, INFO, WARNING, ERROR, or CRITICAL')

    parser.add_argument("--llm.base_url", type=str)
    parser.add_argument("--llm.name", type=str)
    parser.add_argument("--llm.num_ctx", type=int)

    parser.add_argument("--data.load.path", type=str)
    parser.add_argument("--data.load.name", type=str)
    parser.add_argument("--data.load.split", type=str)
    parser.add_argument("--data.task_type", type=str)
    parser.add_argument("--data.topic_path", type=str)
    parser.add_argument("--data.question_path", type=str)
    parser.add_argument("--data.choices_path", type=str)
    parser.add_argument("--data.answer_path", type=str)
    parser.add_argument("--data.num_samples", type=int)

    parser.add_argument("--rag.random_seed", type=int)
    parser.add_argument("--rag.log_every_n_steps", type=int)
    parser.add_argument("--rag.test_mode", type=bool)

    parser.add_argument("--rag.network_type", type=str)
    parser.add_argument("--rag.num_peers", type=int)
    parser.add_argument("--rag.num_peer_attachments", type=int)
    parser.add_argument("--rag.search_algorithm", type=str)
    parser.add_argument("--rag.query_confidence_threshold", type=float)
    parser.add_argument("--rag.num_query_neighbor", type=int)
    parser.add_argument("--rag.query_ttl", type=int)
    parser.add_argument("--rag.filter_out_topic_ratio", type=float)
    parser.add_argument("--rag.filter_out_qa_ratio", type=float)

    parser.add_argument("--config", action="config")
    
    # Add security-related arguments
    parser.add_argument("--security.enable_attack", type=bool)
    parser.add_argument("--security.poisoning_ratio", type=float)
    parser.add_argument("--security.attack_strategy", type=str)
    parser.add_argument("--security.poison_type", type=str)
    parser.add_argument("--security.target_peer_ids", type=list[int])  # For targeted attack
    parser.add_argument("--security.amplification_factor", type=int)  # NEW
    parser.add_argument("--security.question_variants", type=int)     # NEW
    
    # Add Membership Inference Attack arguments
    parser.add_argument("--security.enable_membership_inference", type=bool)
    parser.add_argument("--security.mia_inference_method", type=str)
    parser.add_argument("--security.mia_test_size", type=float)
    parser.add_argument("--security.mia_threshold_percentile", type=int)
    parser.add_argument("--security.mia_random_seed", type=int)

    cfg = parser.parse_args()
    return cfg

