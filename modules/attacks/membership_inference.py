import random
from typing import Dict, Any, List
from loguru import logger
import numpy as np
from sklearn.metrics import roc_auc_score

from modules.attacks.base_attack import BaseAttack
from modules.rag_network import DRAGNetwork
from modules.data_types import Datapoint
from modules.evaluator import QAEvaluator


class MembershipInferenceAttack(BaseAttack):
    """
    Membership Inference Attack on Distributed RAG Systems.
    
    Goal: Determine if a specific data point exists in the knowledge base
    by analyzing retrieval patterns and response confidence.
    """
    
    def __init__(
        self,
        inference_method: str = "confidence_based",
        test_size: float = 0.3,
        threshold_percentile: int = 50,
        random_seed: int = 42
    ):
        super().__init__("MembershipInference")
        self.inference_method = inference_method
        self.test_size = test_size
        self.threshold_percentile = threshold_percentile
        self.random_seed = random_seed
        self.evaluator = QAEvaluator()
    
    def execute(self, network: DRAGNetwork, data_points: List[Datapoint]) -> Dict[str, Any]:
        """Execute membership inference attack."""
        logger.info(f"Executing {self.attack_name} attack...")
        logger.info(f"Inference method: {self.inference_method}")
        logger.info(f"Test size (non-members): {self.test_size}")
        
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        # Split into members (in KB) and non-members (not in KB)
        # 30% of the data points are non-members
        shuffled_data_points = random.sample(data_points, len(data_points))
        num_non_members = max(1, int(len(shuffled_data_points) * self.test_size))
        # 70% of the data points are members
        num_members = len(shuffled_data_points) - num_non_members
        
        if num_members < 1 or num_non_members < 1:
            logger.warning("Not enough data points for MIA. Need at least 1 member and 1 non-member.")
            return self._get_empty_results()
        
        # Select non-members and members
        non_members_data = shuffled_data_points[:num_non_members]
        members_data = shuffled_data_points[num_non_members:]
        
        # Temporarily remove non-members from the network
        original_knowledge_bases = {}
        for dp in non_members_data:
            for peer_id, peer in enumerate(network.peers):
                if dp in peer.knowledge_base.data_points:
                    if peer_id not in original_knowledge_bases:
                        original_knowledge_bases[peer_id] = []
                    original_knowledge_bases[peer_id].append(dp)
                    peer.knowledge_base.data_points.remove(dp)
                    # Rebuild embeddings after removal
                    self._rebuild_embeddings(peer.knowledge_base)
                    break
        
        logger.info(f"Testing with {len(members_data)} members and {len(non_members_data)} non-members")
        
        # Extract features for members
        member_features = []
        logger.info("Extracting features for MEMBERS (data in KB)...")
        for dp in members_data:
            features = self._extract_query_features(network, dp, is_member=True)
            member_features.append(features)
        
        # Extract features for non-members
        non_member_features = []
        logger.info("Extracting features for NON-MEMBERS (data NOT in KB)...")
        for dp in non_members_data:
            features = self._extract_query_features(network, dp, is_member=False)
            non_member_features.append(features)
        
        # Restore original knowledge bases
        for peer_id, dps_to_restore in original_knowledge_bases.items():
            for dp in dps_to_restore:
                network.peers[peer_id].add_knowledge(dp)
        
        # Combine features and labels
        all_features = member_features + non_member_features
        y_true = np.array([f['true_label'] for f in all_features])
        membership_scores = np.array([f['membership_score'] for f in all_features])
        
        # Apply inference method
        threshold = np.percentile(membership_scores, self.threshold_percentile)
        logger.info(f"Inference threshold (percentile {self.threshold_percentile}): {threshold:.4f}")
        y_pred = (membership_scores >= threshold).astype(int)
        
        # Calculate attack success metrics
        metrics = self._calculate_attack_metrics(y_true, y_pred, membership_scores)
        
        logger.info(f"MIA executed: Attack Accuracy = {metrics['attack_accuracy']:.2%}, "
                    f"Privacy Risk = {metrics['privacy_risk']}")
        
        return metrics
    
    def _extract_query_features(self, network: DRAGNetwork, 
                                dp: Datapoint, is_member: bool) -> Dict[str, Any]:
        """Extract features that indicate membership by querying the network."""
        try:
            rag_result = network.topic_query(
                question=dp.question,
                num_query_neighbor=2,
                query_confidence_threshold=0.5,
                max_ttl=6
            )
            rag_answer = rag_result.answer
            num_hops = rag_result.num_hops
            num_messages = rag_result.num_messages
        except Exception as e:
            logger.warning(f"Query failed for '{dp.question[:50]}...': {e}")
            rag_answer = ""
            num_hops = 6
            num_messages = 0
        
        # Calculate semantic similarity
        semantic_similarity = self.evaluator.calculate_semantic_similarity(
            rag_answer, dp.answer
        )
        
        # Calculate other features
        answer_length_ratio = len(rag_answer) / len(dp.answer) if len(dp.answer) > 0 else 0
        max_ttl = 6
        normalized_hops = num_hops / max_ttl if max_ttl > 0 else 1.0
        
        # Compute membership score (higher = more likely to be member)
        membership_score = (
            semantic_similarity * 0.6 +
            (1 - normalized_hops) * 0.3 +
            min(answer_length_ratio, 1.0) * 0.1
        )
        
        features = {
            'true_label': 1 if is_member else 0,
            'semantic_similarity': semantic_similarity,
            'num_hops': num_hops,
            'num_messages': num_messages,
            'answer_length_ratio': answer_length_ratio,
            'membership_score': membership_score
        }
        return features
    
    def _calculate_attack_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  y_scores: np.ndarray) -> Dict[str, Any]:
        """Calculate standard ML attack metrics."""
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        total = tp + tn + fp + fn
        
        accuracy = (tp + tn) / total if total > 0 else 0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tpr
        
        auc_roc = 0.0
        if len(np.unique(y_true)) > 1:
            auc_roc = roc_auc_score(y_true, y_scores)
        
        privacy_risk = self._assess_privacy_risk(accuracy, tpr, auc_roc)
        
        return {
            "confusion_matrix": {'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)},
            "attack_accuracy": float(accuracy),
            "true_positive_rate": float(tpr),
            "false_positive_rate": float(fpr),
            "precision": float(precision),
            "recall": float(recall),
            "auc_roc": float(auc_roc),
            "privacy_risk": privacy_risk,
            "num_members_tested": int(np.sum(y_true == 1)),
            "num_non_members_tested": int(np.sum(y_true == 0))
        }
    
    def _assess_privacy_risk(self, accuracy: float, tpr: float, auc_roc: float) -> str:
        """Assess privacy risk level based on attack accuracy, TPR, and AUC-ROC."""
        risk_score = (accuracy * 0.4) + (tpr * 0.3) + (auc_roc * 0.3)
        
        if risk_score > 0.9:
            return "CRITICAL - Severe privacy leak"
        elif risk_score > 0.75:
            return "HIGH - Significant privacy risk"
        elif risk_score > 0.6:
            return "MEDIUM - Moderate privacy concern"
        elif risk_score > 0.5:
            return "LOW - Slight privacy risk"
        else:
            return "NEGLIGIBLE - Attack ineffective"
    
    def evaluate_success(self, original_metrics: Dict[str, float], 
                         attacked_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Evaluate the success of the attack."""
        attack_accuracy = attacked_metrics.get('attack_accuracy', 0.0)
        privacy_risk = attacked_metrics.get('privacy_risk', 'UNKNOWN')
        
        return {
            "attack_success": attack_accuracy > 0.5,
            "attack_accuracy": attack_accuracy,
            "privacy_risk_level": privacy_risk,
            "interpretation": f"Attack Success: {'YES' if attack_accuracy > 0.5 else 'NO'}. "
                              f"Accuracy: {attack_accuracy:.1%}. "
                              f"Privacy Risk: {privacy_risk}."
        }
    
    def _rebuild_embeddings(self, knowledge_base):
        """Rebuild embeddings after modifying data_points."""
        import torch
        knowledge_base.embeddings = None
        for dp in knowledge_base.data_points:
            formatted_entry = f"{dp.question}"
            new_embedding = knowledge_base.text_embedding_model.encode(
                [formatted_entry],
                convert_to_tensor=True,
                device=knowledge_base.device
            )
            if knowledge_base.embeddings is None:
                knowledge_base.embeddings = new_embedding
            else:
                knowledge_base.embeddings = torch.cat([knowledge_base.embeddings, new_embedding], dim=0)
    
    def _get_empty_results(self) -> Dict[str, Any]:
        """Return empty results if MIA cannot be executed."""
        return {
            "confusion_matrix": {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0},
            "attack_accuracy": 0.0,
            "true_positive_rate": 0.0,
            "false_positive_rate": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "auc_roc": 0.0,
            "privacy_risk": "NEGLIGIBLE - Insufficient data",
            "num_members_tested": 0,
            "num_non_members_tested": 0
        }

