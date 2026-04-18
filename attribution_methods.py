import numpy as np

from config import RANDOM_SEED, compute_direction


class TokenAttributionMethod:
    """Base class for token-level attribution methods."""

    def __init__(self, vectorizer, model, base_seed=RANDOM_SEED):
        self.vectorizer = vectorizer
        self.model = model
        self.base_seed = base_seed

    def get_attribution(self, text, tokens, random_seed=None):
        raise NotImplementedError

    @staticmethod
    def make_occurrence_id(sample_id, token_position):
        return f"{sample_id}:{token_position}"

    def get_occurrence_attribution(
        self,
        text,
        tokens=None,
        sample_id=None,
        random_seed=None,
    ):
        """
        Return one attribution record per token occurrence.

        This keeps repeated token strings separate by using token position and
        occurrence_id instead of deduplicating by token text.
        """
        if not tokens:
            tokens = str(text).split()

        scores = self.get_attribution(
            text=text,
            tokens=tokens,
            random_seed=random_seed,
        )

        return [
            {
                "sample_id": sample_id,
                "token": token,
                "token_position": token_position,
                "occurrence_id": self.make_occurrence_id(sample_id, token_position),
                "attribution_score": float(score),
                "delta_token": float(score),
                "abs_delta_token": float(abs(score)),
                "direction_token": compute_direction(float(score)),
            }
            for token_position, (token, score) in enumerate(zip(tokens, scores))
        ]


class LeaveOneOutAttribution(TokenAttributionMethod):
    """Token-level attribution via deterministic leave-one-out deletion."""

    def get_attribution(self, text, tokens, random_seed=None):
        if not tokens:
            tokens = str(text).split()

        original_score = self.model.predict_proba(self.vectorizer.transform([text]))[
            0, 1
        ]

        scores = []
        for idx in range(len(tokens)):
            text_without_token = " ".join(tokens[:idx] + tokens[idx + 1 :])

            if text_without_token.strip():
                score_without_token = self.model.predict_proba(
                    self.vectorizer.transform([text_without_token])
                )[0, 1]
            else:
                score_without_token = 0.5

            scores.append(float(original_score - score_without_token))

        return scores


class LIMEAttribution(TokenAttributionMethod):
    """Simple seeded LIME-style approximation."""

    def get_attribution(
        self,
        text,
        tokens,
        num_samples=100,
        random_seed=None,
    ):
        if not tokens:
            tokens = str(text).split()

        if not tokens:
            return []

        seed_to_use = self.base_seed if random_seed is None else random_seed
        rng = np.random.default_rng(seed_to_use)

        original_score = self.model.predict_proba(self.vectorizer.transform([text]))[
            0, 1
        ]

        masks = []
        scores = []

        for _ in range(num_samples):
            mask = rng.binomial(1, 0.5, len(tokens))

            if mask.sum() == 0:
                perturbed_text = ""
                score = 0.5
            else:
                perturbed_text = " ".join(
                    token for token, keep in zip(tokens, mask) if keep == 1
                )
                score = self.model.predict_proba(
                    self.vectorizer.transform([perturbed_text])
                )[0, 1]

            masks.append(mask)
            scores.append(score)

        masks = np.asarray(masks)
        scores = np.asarray(scores)
        weights = 1.0 / (1.0 + np.abs(scores - original_score))

        try:
            from sklearn.linear_model import LinearRegression

            reg = LinearRegression()
            reg.fit(masks, scores, sample_weight=weights)
            return reg.coef_.tolist()
        except Exception:
            return [0.0] * len(tokens)


class DBSAAttribution(TokenAttributionMethod):
    """
    Seeded lightweight DBSA-style attribution.

    Output remains a non-directional sensitivity score and is kept mainly for
    compatibility with earlier experiments.
    """

    def __init__(
        self,
        vectorizer,
        model,
        base_seed=RANDOM_SEED,
        n_samples=50,
        noise_scale=0.01,
    ):
        super().__init__(vectorizer, model, base_seed=base_seed)
        self.n_samples = n_samples
        self.noise_scale = noise_scale

    def get_attribution(self, text, tokens, random_seed=None):
        if not tokens:
            tokens = str(text).split()

        seed_to_use = self.base_seed if random_seed is None else random_seed
        rng = np.random.default_rng(seed_to_use)

        original_outputs = self._sample_outputs(text, rng)

        scores = []
        for idx in range(len(tokens)):
            text_without_token = " ".join(tokens[:idx] + tokens[idx + 1 :])
            perturbed_outputs = self._sample_outputs(text_without_token, rng)
            scores.append(self._energy_distance(original_outputs, perturbed_outputs))

        return scores

    def _sample_outputs(self, text, rng):
        if not str(text).strip():
            return np.array([0.5] * self.n_samples).reshape(-1, 1)

        X_dense = self.vectorizer.transform([str(text)]).toarray()
        outputs = []

        for _ in range(self.n_samples):
            noisy = X_dense + rng.normal(0, self.noise_scale, X_dense.shape)
            outputs.append(float(self.model.predict_proba(noisy)[0, 1]))

        return np.asarray(outputs).reshape(-1, 1)

    @staticmethod
    def _energy_distance(dist1, dist2):
        from scipy.spatial.distance import cdist, pdist

        d1_intra = np.mean(pdist(dist1)) if len(dist1) > 1 else 0.0
        d2_intra = np.mean(pdist(dist2)) if len(dist2) > 1 else 0.0
        d_inter = np.mean(cdist(dist1, dist2))
        return float(max(0.0, 2 * d_inter - d1_intra - d2_intra))


def get_attribution_method(
    method_name,
    vectorizer,
    model,
    base_seed=RANDOM_SEED,
):
    method = method_name.lower()

    if method == "leave_one_out":
        return LeaveOneOutAttribution(
            vectorizer=vectorizer,
            model=model,
            base_seed=base_seed,
        )
    if method == "lime":
        return LIMEAttribution(
            vectorizer=vectorizer,
            model=model,
            base_seed=base_seed,
        )
    if method == "dbsa":
        return DBSAAttribution(
            vectorizer=vectorizer,
            model=model,
            base_seed=base_seed,
        )

    raise ValueError(f"Unknown attribution method: {method_name}")
