#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hash-based Experiment Design Tool

A deterministic and non-manipulable sample allocation system for biomedical research,
using cryptographic hashing for group assignment. Ensures reproducible and balanced
group allocations while preventing selection bias.

Features:
- SHA-256 with salt for secure hashing
- Stratified sampling support
- Automatic balance validation
- Type annotations and full documentation

Author: Gloria
Date: 2025-03-11
License: MIT
"""

import hashlib
import numpy as np
from typing import List, Dict, Any, Union, Optional
from collections import defaultdict, Counter


class HashExperimentDesigner:
    """
    A cryptographically secure experiment designer for balanced sample allocation.

    Attributes:
        n_groups (int): Number of experimental groups (≥2)
        seed (int): Seed for salt generation (ensures different allocations)
        salt (bytes): Cryptographic salt generated from seed
        _hash_algo (str): Hash algorithm used (default: sha256)

    Raises:
        ValueError: Invalid parameters
        TypeError: Incorrect parameter types
    """

    def __init__(self, n_groups: int = 2, seed: int = 42, hash_algo: str = "sha256"):
        """
        Initialize the allocation system.

        Args:
            n_groups: Number of experimental groups (≥2)
            seed: Seed for reproducible salt generation
            hash_algo: Hash algorithm (default: sha256)

        Raises:
            ValueError: If n_groups < 2 or invalid hash algorithm
            TypeError: If parameters have incorrect types
        """
        # Type validation
        if not isinstance(n_groups, int):
            raise TypeError(f"n_groups must be integer, got {type(n_groups)}")
        if not isinstance(seed, int):
            raise TypeError(f"seed must be integer, got {type(seed)}")

        # Value validation
        if n_groups < 2:
            raise ValueError(f"n_groups must be ≥2, got {n_groups}")
        if hash_algo not in hashlib.algorithms_available:
            raise ValueError(f"Unsupported hash algorithm: {hash_algo}")

        self.n_groups = n_groups
        self.seed = seed
        self._hash_algo = hash_algo

        # Generate salt using seeded randomness
        np.random.seed(seed)
        self.salt = np.random.bytes(16)  # 128-bit salt

    def _hash_sample(self, sample_id: Union[str, int]) -> int:
        """
        Compute salted hash of sample ID.

        Args:
            sample_id: Unique sample identifier

        Returns:
            int: 64-bit integer representation of the hash

        Raises:
            TypeError: If sample_id cannot be converted to string
        """
        try:
            str_id = str(sample_id)
        except Exception as e:
            raise TypeError(f"sample_id must be string-convertible: {e}") from None

        hasher = hashlib.new(self._hash_algo)
        hasher.update(str_id.encode("utf-8"))
        hasher.update(self.salt)
        
        # Use first 8 bytes (64 bits) of digest
        return int.from_bytes(hasher.digest()[:8], byteorder="big")

    def assign_group(self, sample_id: Union[str, int]) -> int:
        """
        Assign a sample to an experimental group.

        Args:
            sample_id: Unique sample identifier

        Returns:
            int: Group ID between 0 and n_groups-1
        """
        hash_value = self._hash_sample(sample_id)
        return hash_value % self.n_groups

    def assign_samples(self, sample_ids: List[Union[str, int]]) -> Dict[int, List[Union[str, int]]]:
        """
        Assign multiple samples to experimental groups.

        Args:
            sample_ids: List of sample identifiers

        Returns:
            Dict[int, List]: {group_id: [sample_ids]}

        Raises:
            ValueError: If sample_ids is empty
        """
        if not sample_ids:
            raise ValueError("Cannot assign empty sample list")

        assignments = defaultdict(list)
        for sid in sample_ids:
            group = self.assign_group(sid)
            assignments[group].append(sid)
        
        return dict(assignments)

    @staticmethod
    def validate_balance(assignments: Dict[int, List[Any]]) -> Dict[str, float]:
        """
        Analyze group balance statistics.

        Args:
            assignments: {group_id: [samples]}

        Returns:
            Dict[str, float]: Balance metrics:
                - mean: Average group size
                - std: Standard deviation
                - min: Smallest group size
                - max: Largest group size
                - imbalance_ratio: (max-min)/mean

        Raises:
            ValueError: If assignments is empty
        """
        if not assignments:
            raise ValueError("Empty assignments")

        counts = [len(v) for v in assignments.values()]
        mean = np.mean(counts)
        std = np.std(counts)
        min_count = min(counts)
        max_count = max(counts)
        imbalance = (max_count - min_count) / mean if mean != 0 else 0.0

        return {
            "mean": mean,
            "std": std,
            "min": min_count,
            "max": max_count,
            "imbalance_ratio": imbalance,
        }

    def stratified_assign(
        self,
        sample_ids: List[Union[str, int]],
        strata: Dict[Union[str, int], Any]
    ) -> Dict[int, List[Union[str, int]]]:
        """
        Perform stratified group assignment.

        Args:
            sample_ids: List of sample identifiers
            strata: {sample_id: stratum} mapping

        Returns:
            Dict[int, List]: {group_id: [sample_ids]}

        Raises:
            ValueError: If samples missing from strata
        """
        # Validate strata coverage
        missing = set(sample_ids) - set(strata.keys())
        if missing:
            raise ValueError(f"Samples missing strata: {missing}")

        # Organize by stratum
        stratum_map = defaultdict(list)
        for sid in sample_ids:
            stratum_map[strata[sid]].append(sid)

        # Allocate per stratum
        assignments = defaultdict(list)
        for stratum_samples in stratum_map.values():
            stratum_assign = self.assign_samples(stratum_samples)
            for group, samples in stratum_assign.items():
                assignments[group].extend(samples)

        return dict(assignments)


if __name__ == "__main__":
    """Example usage with validation"""
    # Basic allocation
    designer = HashExperimentDesigner(n_groups=3)
    samples = [f"SAMPLE_{i}" for i in range(100)]
    
    print("=== Basic Allocation ===")
    assignments = designer.assign_samples(samples)
    stats = designer.validate_balance(assignments)
    print(f"Groups: {[len(v) for v in assignments.values()]}")
    print(f"Imbalance: {stats['imbalance_ratio']:.1%}")

    # Stratified allocation
    print("\n=== Stratified Allocation ===")
    strata = {s: "A" if i % 3 == 0 else "B" for i, s in enumerate(samples)}
    stratified = designer.stratified_assign(samples, strata)
    
    # Analyze stratum distribution
    print("\nStratum distribution per group:")
    for group, samples in stratified.items():
        cnt = Counter(strata[s] for s in samples)
        print(f"Group {group}: {dict(cnt)}")


# In[ ]:




