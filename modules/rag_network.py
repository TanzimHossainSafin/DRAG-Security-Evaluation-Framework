from collections import deque
import random
from typing import Dict, List, Optional

from loguru import logger
import networkx as nx
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from modules.data_types import RAGAnswer, Datapoint
from modules.peer import Peer


class DRAGNetwork:
    def __init__(
            self, 
            num_peers: int, 
            num_peer_attachments: int, 
            llm_url: str, 
            llm_name: str, 
            llm_num_ctx: int,
            llm_seed: int, 
            embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
        ):
        self.num_peers = num_peers
        self.network = nx.barabasi_albert_graph(num_peers, num_peer_attachments)
        self.text_embedding_model = SentenceTransformer(embedding_model)
        self.peers = [Peer(peer_id, llm_url, llm_name, llm_num_ctx, llm_seed, self.text_embedding_model) 
                      for peer_id in range(num_peers)]

        self.peer_topics: Dict[int, List[str]] = {peer_id: [] for peer_id in range(self.num_peers)}
        self.topic_peers: Dict[str, List[int]] = {}
        self.all_topics: List[str] = []

    def init_knowledge(self, data_points: List[Datapoint]):
        """
        Distributes data points to peers based on their topics (Uniform).

        Args:
            data_points: A list of Datapoint objects.
        """
        # Count all topics
        topic_check: Dict[str, bool] = {}
        for data_point in data_points:
            topic_check[data_point.topic] = True
        self.all_topics = list(topic_check.keys())

        # Assign topics to peers
        self.topic_peers = {topic: [] for topic in self.all_topics}
        for topic in self.all_topics:
            peer_id = random.choice(range(self.num_peers))
            self.topic_peers[topic].append(peer_id)
            self.peer_topics[peer_id].append(topic)

        # Distribute data points to peers based on assigned topics
        for data_point in tqdm(data_points, desc=f"Distributing data points to peers"):
            for peer_id in self.topic_peers[data_point.topic]:
                self.peers[peer_id].add_knowledge(data_point)

    def topic_query(
            self, 
            question: str, 
            query_peer_id: Optional[int] = None, 
            num_query_neighbor: int = 2,
            query_confidence_threshold: float = 0.5,
            max_ttl: int = 6
    ) -> RAGAnswer:
        """
        Topic-based network query

        Args:
            question: The question to ask.
            query_peer_id: The ID of the peer initiating the query. If None, a random peer is selected.
            num_query_neighbor: The maximum number of neighbors to query at each hop.
            query_confidence_threshold: The confidence threshold required for an answer to be accepted.
            max_ttl: The time-to-live for the query (maximum number of hops).

        Returns:
            The answer to the question if found, otherwise None.
        """
        # Track number of messages (queries) sent
        num_messages = 0

        # Initialize query peer
        if query_peer_id is None:
            query_peer_id = random.choice(range(self.num_peers))
            logger.debug(f"Randomly selected starting peer: {query_peer_id}")
        
        logger.debug(f"Starting topic-based search from peer {query_peer_id}")
        logger.debug(f"Parameters: num_neighbors={num_query_neighbor}, max_ttl={max_ttl}")

        # Determine the topic of the question first
        question_topic = self.peers[query_peer_id].parse_topic(question, self.all_topics)
        logger.debug(f"Parsed question topic: {question_topic}")

        # Keep track of visited peers to avoid cycles
        visited_ids = {query_peer_id}

        # Queue for BFS: (peer_id, hop)
        queue = deque([(query_peer_id, 0)])

        # Perform topic-based search
        while queue:
            current_peer_id, hop = queue.popleft()

            if hop >= max_ttl:
                continue

            logger.debug(f"Hop {hop}: Querying peer {current_peer_id}")

            # Query the current peer and track message
            current_answer, relevant_knowledge, relevant_score, is_query_hit = \
                self.peers[current_peer_id].query(question, query_confidence_threshold)
            num_messages += 1

            if current_answer is not None:
                logger.debug(f"Answer found at peer {current_peer_id} after {hop} hops")
                logger.debug(f"Total messages sent: {num_messages}")
                answer = RAGAnswer(
                    answer=str(current_answer),
                    relevant_knowledge=relevant_knowledge,
                    relevant_score=relevant_score,
                    num_hops=hop,
                    num_messages=num_messages,
                    is_query_hit=is_query_hit
                )
                return answer  # Return the answer if found

            # Get and prioritize neighbors
            current_neighbor_ids = list(self.network.neighbors(current_peer_id))
            picked_neighbor_ids = []

            if len(current_neighbor_ids) > num_query_neighbor:
                # Find topic-matched neighbors
                topic_matched_neighbors = []
                for neighbor_id in current_neighbor_ids:
                    if question_topic in self.peer_topics[neighbor_id]:
                        topic_matched_neighbors.append(neighbor_id)

                # Add topic-matched peers to the neighbor list of current peer
                # So that in the future, there is no need to flood other peers
                logger.debug(f"Found {len(topic_matched_neighbors)} topic-matched neighbors, update neighbor list")
                for neighbor_id in topic_matched_neighbors:
                    self.network.add_edge(query_peer_id, neighbor_id)

                # Select neighbors based on topic matching
                if len(topic_matched_neighbors) > num_query_neighbor:
                    picked_neighbor_ids = random.sample(topic_matched_neighbors, num_query_neighbor)
                else:
                    picked_neighbor_ids = topic_matched_neighbors
                    # Fill remaining slots with random neighbors
                    remaining_neighbor_ids = list(set(current_neighbor_ids) - set(picked_neighbor_ids))
                    remaining_num = min(num_query_neighbor - len(picked_neighbor_ids), len(remaining_neighbor_ids))
                    picked_neighbor_ids += random.sample(remaining_neighbor_ids, remaining_num)
            else:
                picked_neighbor_ids = current_neighbor_ids

            logger.debug(f"Selected neighbors for next hop: {picked_neighbor_ids}")

            # Add picked neighbors to the queue
            for neighbor_id in picked_neighbor_ids:
                if neighbor_id not in visited_ids:
                    visited_ids.add(neighbor_id)
                    queue.append((neighbor_id, hop + 1))
                    logger.debug(f"Added neighbor {neighbor_id} to queue at Hop {hop + 1}")

        logger.debug(f"Search failed after {max_ttl} hops")
        logger.debug(f"Total messages sent: {num_messages}")

        # Return empty answer if no result found
        answer = RAGAnswer(
            answer="",
            relevant_knowledge="",
            relevant_score=0.0,
            num_hops=max_ttl,
            num_messages=num_messages,
            is_query_hit=False
        )
        return answer

    def random_walk_query(
            self,
            question: str,
            query_peer_id: Optional[int] = None,
            query_confidence_threshold: float = 0.5,
            max_ttl: int = 6,
            restart_probability: float = 0.1
    ) -> RAGAnswer:
        """
        Queries the network using a random walk algorithm with restart probability.
        
        Args:
            question: The question to ask
            query_peer_id: Starting peer ID (random if None)
            query_confidence_threshold: Minimum confidence required for an answer
            max_ttl: Maximum number of steps in the walk
            restart_probability: Probability of restarting from the initial peer
        
        Returns:
            RAGAnswer object containing the response
        """
        # Track number of messages (queries) sent
        num_messages = 0

        # Initialize starting peer
        if query_peer_id is None:
            query_peer_id = random.choice(range(self.num_peers))
            logger.debug(f"Randomly selected starting peer: {query_peer_id}")
        
        current_peer_id = query_peer_id
        initial_peer_id = query_peer_id
        
        # Keep track of path for debugging
        path = [current_peer_id]

        logger.debug(f"Starting random walk search from peer {initial_peer_id}")
        logger.debug(f"Parameters: max_ttl={max_ttl}, restart_prob={restart_probability}")
        
        # Perform random walk
        for hop in range(max_ttl):
            logger.debug(f"Hop {hop}: Querying peer {current_peer_id}")
            
            # Query current peer
            current_answer, relevant_knowledge, relevant_score, is_query_hit = \
                self.peers[current_peer_id].query(question, query_confidence_threshold)
            num_messages += 1
            
            # Return if answer found
            if current_answer is not None:
                logger.debug(f"Answer found at peer {current_peer_id} after {hop} hops")
                logger.debug(f"Total messages sent: {num_messages}")
                return RAGAnswer(
                    answer=str(current_answer),
                    relevant_knowledge=relevant_knowledge,
                    relevant_score=relevant_score,
                    num_hops=hop,
                    num_messages=num_messages,
                    is_query_hit=is_query_hit
                )
            
            # Decide whether to restart
            if random.random() < restart_probability:
                current_peer_id = initial_peer_id
                logger.debug(f"Random restart to initial peer: {initial_peer_id}")
            else:
                # Get neighbors and randomly select next peer
                neighbors = list(self.network.neighbors(current_peer_id))
                if not neighbors:
                    logger.debug(f"Dead end at peer {current_peer_id}, restarting")
                    current_peer_id = initial_peer_id
                else:
                    current_peer_id = random.choice(neighbors)
                    logger.debug(f"Moving to neighbor peer {current_peer_id}")
            
            path.append(current_peer_id)
        
        logger.debug(f"Search failed after {max_ttl} hops")
        logger.debug(f"Path taken: {path}")
        logger.debug(f"Total messages sent: {num_messages}")
        
        # Return empty answer if no result found
        return RAGAnswer(
            answer="",
            relevant_knowledge="",
            relevant_score=0.0,
            num_hops=max_ttl,
            num_messages=num_messages,
            is_query_hit=False
        )

    def flooding_query(
            self,
            question: str,
            query_peer_id: Optional[int] = None,
            query_confidence_threshold: float = 0.5,
            max_ttl: int = 6
    ) -> RAGAnswer:
        """
        Queries the network using a flooding algorithm.
        
        Args:
            question: The question to ask
            query_peer_id: Starting peer ID (random if None)
            query_confidence_threshold: Minimum confidence required for an answer
            max_ttl: Maximum time-to-live (network depth to explore)
        
        Returns:
            RAGAnswer object containing the response
        """
        # Track number of messages (queries) sent
        num_messages = 0

        # Initialize starting peer
        if query_peer_id is None:
            query_peer_id = random.choice(range(self.num_peers))
            logger.debug(f"Randomly selected starting peer: {query_peer_id}")
        
        logger.debug(f"Starting flooding search from peer {query_peer_id}")
        logger.debug(f"Parameters: max_ttl={max_ttl}")
        
        # Use set for visited nodes to avoid cycles
        visited_ids = {query_peer_id}
        
        # Queue for BFS: (peer_id, hop)
        queue = deque([(query_peer_id, 0)])
        
        while queue:
            current_peer_id, hop = queue.popleft()
            
            if hop >= max_ttl:
                continue

            logger.debug(f"Flooding at Hop {hop}, peer: {current_peer_id}")
            
            # Query current peer and track message
            current_answer, relevant_knowledge, relevant_score, is_query_hit = \
                self.peers[current_peer_id].query(question, query_confidence_threshold)
            num_messages += 1
            
            # Return if answer found
            if current_answer is not None:
                logger.debug(f"Answer found at peer {current_peer_id}, Hop {hop}")
                logger.debug(f"Total messages sent: {num_messages}")
                return RAGAnswer(
                    answer=str(current_answer),
                    relevant_knowledge=relevant_knowledge,
                    relevant_score=relevant_score,
                    num_hops=hop,
                    num_messages=num_messages,
                    is_query_hit=is_query_hit
                )
            
            # Add all unvisited neighbors to queue
            neighbors = self.network.neighbors(current_peer_id)
            for neighbor_id in neighbors:
                if neighbor_id not in visited_ids:
                    visited_ids.add(neighbor_id)
                    queue.append((neighbor_id, hop + 1))
                    logger.debug(f"Added neighbor {neighbor_id} to queue at Hop {hop + 1}")
        
        # Log search statistics
        logger.debug("Search failed to find answer")
        logger.debug(f"Total messages sent: {num_messages}")
        
        # Return empty answer if no result found
        return RAGAnswer(
            answer="",
            relevant_knowledge="",
            relevant_score=0.0,
            num_hops=max_ttl,
            num_messages=num_messages,
            is_query_hit=False
        )


class CRAGNetwork:
    def __init__(
            self, 
            llm_url: str, 
            llm_name: str, 
            llm_num_ctx: int,
            llm_seed: int, 
            embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
        ):
        self.text_embedding_model = SentenceTransformer(embedding_model)
        self.peer = Peer(0, llm_url, llm_name, llm_num_ctx, llm_seed, self.text_embedding_model)

    def init_knowledge(self, data_points: List[Datapoint]):
        """
        Distributes data points to peers based on their topics (Uniform).

        Args:
            data_points: A list of Datapoint objects.
        """
        # Distribute data points to peers based on assigned topics
        for data_point in tqdm(data_points, desc=f"Distributing data points to peers"):
            self.peer.add_knowledge(data_point)

    def query(
            self,
            question: str,
            query_confidence_threshold: float = 0.5
    ) -> RAGAnswer:
        """
        Queries the network using a random walk algorithm with restart probability.
        
        Args:
            question: The question to ask
            query_confidence_threshold: Minimum confidence required for an answer
        
        Returns:
            RAGAnswer object containing the response
        """
        # Query current peer
        current_answer, relevant_knowledge, relevant_score, is_query_hit = \
            self.peer.query(question, query_confidence_threshold)

        # Return if answer found
        if current_answer is not None:
            logger.debug(f"Answer found at peer 0")
            return RAGAnswer(
                answer=str(current_answer),
                relevant_knowledge=relevant_knowledge,
                relevant_score=relevant_score,
                num_hops=0,
                num_messages=0,
                is_query_hit=is_query_hit
            )
            
        logger.debug(f"Search failed")
        
        # Return empty answer if no result found
        return RAGAnswer(
            answer="",
            relevant_knowledge="",
            relevant_score=0.0,
            num_hops=0,
            num_messages=0,
            is_query_hit=False
        )


class NoRAGNetwork:
    def __init__(
            self, 
            llm_url: str, 
            llm_name: str, 
            llm_num_ctx: int,
            llm_seed: int
        ):
        self.peer = Peer(0, llm_url, llm_name, llm_num_ctx, llm_seed)

    def init_knowledge(self, data_points: List[Datapoint]):
        """
        No need to initialize knowledge for No RAG mode

        Args:
            data_points: A list of Datapoint objects.
        """
        pass

    def query(
            self,
            question: str
    ) -> RAGAnswer:
        """
        Queries the network using a random walk algorithm with restart probability.

        Args:
            question: The question to ask
            query_confidence_threshold: Minimum confidence required for an answer

        Returns:
            RAGAnswer object containing the response
        """
        # Query current peer
        generated_answer, relevant_knowledge, relevant_score, is_query_hit = \
            self.peer.query_no_rag(question)

        # Return empty answer if no result found
        return RAGAnswer(
            answer=str(generated_answer),
            relevant_knowledge=relevant_knowledge,
            relevant_score=relevant_score,
            num_hops=0,
            num_messages=0,
            is_query_hit=is_query_hit
        )
