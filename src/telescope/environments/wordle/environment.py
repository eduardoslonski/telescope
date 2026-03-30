"""
Wordle Environment — lightweight implementation using TextArena for the dictionary.

Adapted from https://app.primeintellect.ai/dashboard/environments/will/wordle

Uses TextArena once at init to extract the word list, validation dictionary,
and initial prompt, then discards the heavy env. Per-game state is a tiny
WordleGame dataclass — no deepcopy needed.

Requires: uv add textarena
"""

REQUIRED_PACKAGES = ["textarena"]

import logging
import random
import re
from dataclasses import dataclass, field

from telescope.environments.base import (
    MultiTurnEnvironment,
    Sample,
    RolloutState,
    ChatMessage,
)
from telescope.environments.rewards import Rubric

logger = logging.getLogger(__name__)


DEFAULT_SYSTEM_PROMPT = """You are a competitive game player. \
Make sure you read the game instructions carefully, and always follow the required format.

In each turn, think step-by-step, then give your guess inside <guess>...</guess> tags."""


# ---------------------------------------------------------------------------
# Lightweight per-game state (no heavy objects — safe to create thousands)
# ---------------------------------------------------------------------------

@dataclass
class WordleGame:
    """
    Minimal Wordle game state.  Only stores primitives and small lists.
    The heavy word-list / dictionary live on the *environment* and are shared.
    """
    secret_word: str
    word_length: int = 5
    num_guesses: int = 6
    guess_history: list = field(default_factory=list)   # [(word, [feedback])]
    done: bool = False
    turn: int = 0
    error_count: int = 0
    error_allowance: int = 1
    game_info: dict = field(default_factory=lambda: {
        0: {"role": "Player 0", "invalid_move": False, "turn_count": 0}
    })

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, action: str, is_valid_word: callable) -> tuple[bool, str]:
        """
        Process one guess.

        Args:
            action: raw string from the model (expected to contain ``[word]``).
            is_valid_word: callback ``(str) -> bool`` using the shared dictionary.

        Returns:
            ``(done, observation_text)``  where *observation_text* is the
            feedback string in the same format TextArena + LLMObservationWrapper
            would produce.  Empty string when the game is over.
        """
        # --- parse brackets (matches TextArena WordleEnv.step regex) ---
        match = re.search(r"\[(\w+)\]", action)
        if match is None:
            return self._handle_invalid(
                "You tried submitting a word in the wrong format. "
                "Please make sure to use squared brackets."
            )

        word = match.group(1).lower()

        # --- validate length ---
        if len(word) != self.word_length:
            return self._handle_invalid(
                f"Your word must be exactly {self.word_length} letters."
            )

        # --- duplicate check ---
        if word in [w for w, _ in self.guess_history]:
            return self._handle_invalid(
                f"You have already guessed '{word}' before. "
                "Please try a different word."
            )

        # --- dictionary check ---
        if not is_valid_word(word):
            return self._handle_invalid(f"'{word}' is not an English word.")

        # --- valid move: reset error counter (matches SinglePlayerState) ---
        self.error_count = 0
        self.turn += 1
        self.game_info[0]["turn_count"] += 1

        # --- evaluate ---
        feedback = self._evaluate_guess(word)
        self.guess_history.append((word, feedback))

        # --- win? ---
        if all(f == "G" for f in feedback):
            self.done = True
            self.game_info[0]["reason"] = (
                "Congratulations! You guessed the word correctly!"
            )
            return True, ""

        # --- build observation (same format as TextArena GAME_MESSAGE) ---
        player_view = self._render_player_view()
        guesses_left = self.num_guesses - len(self.guess_history)
        observation = (
            f"You submitted [{word}].\n"
            f"Feedback:\n{player_view}\n"
            f"You have {guesses_left} guesses left."
        )

        # --- max guesses reached? ---
        if len(self.guess_history) >= self.num_guesses:
            self.done = True
            pct = self._get_percentage_completion()
            self.game_info[0]["reason"] = (
                f"The turn limit has been reached. You didn't guess the word, "
                f"but your best guess matched {round(pct * 100)}% of the letters "
                f"in the correct positions.\n"
                f"The secret word was: **{self.secret_word}**."
            )
            return True, ""

        return False, observation

    # ------------------------------------------------------------------
    # Invalid-move handling (mirrors SinglePlayerState.set_invalid_move)
    # ------------------------------------------------------------------

    def _handle_invalid(self, reason: str) -> tuple[bool, str]:
        if self.error_allowance > self.error_count:
            self.error_count += 1
            msg = (
                f"You attempted an invalid move. Reason: {reason} "
                "Please resubmit a valid move and remember to follow "
                "the game rules to avoid penalties."
            )
            return False, msg
        else:
            # too many consecutive errors → game over
            pct = self._get_percentage_completion()
            self.done = True
            self.game_info[0]["reason"] = f"Invalid Move: {reason}"
            self.game_info[0]["invalid_move"] = True
            return True, ""

    # ------------------------------------------------------------------
    # Guess evaluation (identical to TextArena WordleEnv._evaluate_guess)
    # ------------------------------------------------------------------

    def _evaluate_guess(self, guess: str) -> list[str]:
        feedback = [None] * self.word_length
        secret_list = list(self.secret_word)
        guess_list = list(guess)

        # first pass: greens
        for i in range(self.word_length):
            if guess_list[i] == secret_list[i]:
                feedback[i] = "G"
                secret_list[i] = None

        # second pass: yellows / grays
        for i in range(self.word_length):
            if feedback[i] is None:
                if guess_list[i] in secret_list:
                    feedback[i] = "Y"
                    secret_list[secret_list.index(guess_list[i])] = None
                else:
                    feedback[i] = "X"

        return feedback

    # ------------------------------------------------------------------
    # Rendering helpers (identical to TextArena WordleEnv)
    # ------------------------------------------------------------------

    def _render_player_view(self) -> str:
        if not self.guess_history:
            return "No guesses yet."
        word, feedback = self.guess_history[-1]
        return f"{' '.join(word.upper())}\n{' '.join(feedback)}"

    def _get_percentage_completion(self) -> float:
        if not self.guess_history:
            return 0.0
        _, latest = self.guess_history[-1]
        greens = sum(1 for f in latest if f == "G")
        yellows = sum(1 for f in latest if f == "Y") * 0.5
        return (greens + yellows) / self.word_length


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class WordleEnvironment(MultiTurnEnvironment):
    """
    Multi-turn Wordle game environment (TextArena-free).

    Heavy state (word list, dictionary) is loaded once and shared.
    Per-game state is a tiny ``WordleGame`` dataclass — no copy needed.

    Reward structure:
    - correct_answer (weight=1.0): 1.0 if parsed <guess> matches [answer]
    - partial_answer (weight=1.0): Partial credit based on G/Y in feedback
    - length_bonus  (weight=1.0): 1/num_guesses if correct
    - format_reward  (weight=0.2): XML format quality of <guess> tags
    """

    def __init__(
        self,
        max_turns: int = 6,
        num_samples: int = 2000,
        seed: int = 0,
        system_prompt: str | None = None,
        **kwargs,
    ):
        super().__init__(
            max_turns=max_turns,
            system_prompt=system_prompt or DEFAULT_SYSTEM_PROMPT,
            **kwargs,
        )
        self.num_samples_config = num_samples
        self.seed = seed

        # Loaded once in _init_words(), shared across all games
        self._word_list: list[str] | None = None
        self._valid_words: set[str] | None = None
        self._initial_observation: str | None = None

        self.rubric = Rubric()
        self.rubric.add_reward(self._reward_correct, range_min=0, range_max=1)
        self.rubric.add_reward(self._reward_partial, range_min=0, range_max=1)
        self.rubric.add_reward(self._reward_length, range_min=0, range_max=1)
        self.rubric.add_reward(self._reward_fmt, name="format_reward", weight=0.2, range_min=0, range_max=1)

    @property
    def metrics_ranges(self):
        return self.rubric.metrics_ranges

    # ------------------------------------------------------------------
    # One-time initialisation
    # ------------------------------------------------------------------

    def _init_words(self):
        """Load word list and dictionary from TextArena *once*.  Shared across all games.

        Creates a temporary TextArena Wordle-v0 env to extract the word list,
        validation dictionary, and initial prompt (matching prime-envs/wordle
        exactly), then discards the heavy env.  Only lightweight list/set are
        kept — no deepcopy ever needed.
        """
        if self._word_list is not None:
            return

        try:
            import textarena as ta  # type: ignore
        except ImportError as e:
            raise ImportError(
                "WordleEnvironment requires textarena. "
                "Install with: uv add textarena"
            ) from e

        # Create a temporary TextArena env to extract word data + prompt
        ta_env = ta.make(env_id="Wordle-v0")
        ta_env.reset(num_players=1)

        # Word list (guessable secret words) — directly from TextArena
        self._word_list = list(ta_env.word_list)

        # Validation dictionary — from TextArena's EnglishDictionary
        # (uk_words | us_words | nltk_words)
        self._valid_words = {w.lower() for w in ta_env.dictionary.get_all_words()}
        self._valid_words.update(w.lower() for w in self._word_list)

        # Initial observation — from TextArena's actual prompt
        _, self._initial_observation = ta_env.get_observation()

        # Discard the heavy TextArena env — only keep the extracted data
        del ta_env

        logger.info(
            f"Wordle word list loaded: {len(self._word_list)} words, "
            f"dictionary: {len(self._valid_words)} words"
        )

    def _is_valid_word(self, word: str) -> bool:
        """Check if *word* is a valid English word (shared dictionary)."""
        return word.lower() in self._valid_words

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    def load_dataset(self, num_samples: int = -1, **kwargs) -> list[Sample]:
        """Generate Wordle episodes with random secret words."""
        self._init_words()

        n = num_samples if num_samples > 0 else self.num_samples_config
        random.seed(self.seed)

        samples = []
        for _ in range(n):
            secret_word = random.choice(self._word_list)
            samples.append(Sample(
                prompt=self._initial_observation,
                answer=secret_word,
                metadata={"secret_word": secret_word},
            ))

        self._samples = samples
        return samples

    # ------------------------------------------------------------------
    # Rollout lifecycle
    # ------------------------------------------------------------------

    def create_initial_state(self, sample: Sample) -> RolloutState:
        """Create state with a lightweight WordleGame — no deepcopy!"""
        self._init_words()

        state = super().create_initial_state(sample)
        state.custom["game"] = WordleGame(secret_word=sample.answer)
        return state

    # ------------------------------------------------------------------
    # Guess parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_guess(content: str) -> str | None:
        """Extract guess from ``<guess>...</guess>`` tags."""
        match = re.search(r"<guess>\s*(.*?)\s*</guess>", content, re.DOTALL)
        return match.group(1).strip() if match else None

    # ------------------------------------------------------------------
    # Multi-turn interaction
    # ------------------------------------------------------------------

    async def env_response(
        self,
        messages: list[ChatMessage],
        state: RolloutState,
    ) -> list[ChatMessage]:
        """
        Process model's guess and return Wordle feedback.

        - Parses guess from <guess> tags
        - Sends raw parsed string to the game (including brackets if present)
        - Returns empty list when game ends to stop the rollout
        """
        game: WordleGame = state.custom["game"]

        # Extract guess from last assistant message
        guess = None
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                guess = self._parse_guess(msg.get("content", ""))
                break

        logger.debug(f"Parsed {guess=}")

        # Submit guess (str(None) = "None" for consistency)
        done, observation = game.step(str(guess), self._is_valid_word)

        if done:
            logger.debug(f"Game completed! {game.game_info=}")
            state.custom["game_info"] = game.game_info
            return []

        # Extract feedback portion (mirrors _parse_feedback on TextArena obs)
        if "Feedback:" in observation:
            feedback = observation.split("Feedback:")[-1]
        else:
            feedback = observation

        logger.debug(f"Parsed feedback={feedback!r}")

        return [{"role": "user", "content": str(feedback)}]

    # ------------------------------------------------------------------
    # Stop condition
    # ------------------------------------------------------------------

    def is_done(self, state: RolloutState) -> tuple[bool, str | None]:
        """
        Game-over from the game engine is handled by env_response returning []
        (empty), which the orchestrator catches as 'empty_env_response'.
        This only provides the max_turns safety net.
        """
        if self.max_turns > 0 and state.num_turns >= self.max_turns:
            return True, "max_turns_reached"
        return False, None

    # ------------------------------------------------------------------
    # Reward helpers (identical to original WordleEnvironment)
    # ------------------------------------------------------------------

    def _get_all_messages(self, state: RolloutState) -> list[ChatMessage]:
        if not state.trajectory:
            return []
        last_step = state.trajectory[-1]
        messages: list[ChatMessage] = []
        if isinstance(last_step.prompt, list):
            messages.extend(last_step.prompt)
        if isinstance(last_step.completion, list):
            messages.extend(last_step.completion)
        elif isinstance(last_step.completion, str):
            messages.append({"role": "assistant", "content": last_step.completion})
        return messages

    @staticmethod
    def _get_assistant_messages(messages: list[ChatMessage]) -> list[ChatMessage]:
        return [m for m in messages if m.get("role") == "assistant"]

    @staticmethod
    def _get_user_messages(messages: list[ChatMessage]) -> list[ChatMessage]:
        return [m for m in messages if m.get("role") == "user"]

    def _reward_correct_answer(self, messages: list[ChatMessage], answer: str) -> float:
        guess = None
        for msg in reversed(self._get_assistant_messages(messages)):
            parsed = self._parse_guess(str(msg.get("content", "")))
            if parsed is not None:
                guess = parsed
                break
        return 1.0 if guess == "[" + answer + "]" else 0.0

    def _reward_partial_answer(self, messages: list[ChatMessage], answer: str) -> float:
        if self._reward_correct_answer(messages, answer):
            return 0.0
        for user_msg in reversed(self._get_user_messages(messages)):
            feedback = user_msg["content"].strip()
            parts = feedback.split("\n")
            if len(parts) == 3:
                _, scoring, _ = parts
                scoring = scoring.strip()
                return 0.2 * scoring.count("G") + 0.1 * scoring.count("Y")
        return 0.0

    def _reward_length_bonus(self, messages: list[ChatMessage], answer: str) -> float:
        guesses = [
            m for m in self._get_assistant_messages(messages)
            if re.search(r"<guess>.*</guess>", m.get("content", ""))
        ]
        is_correct = self._reward_correct_answer(messages, answer)
        return is_correct / (len(guesses) or 1)

    def _reward_format(self, messages: list[ChatMessage]) -> float:
        model_messages = self._get_assistant_messages(messages)
        if not model_messages:
            return 0.0

        scores = []
        for msg in model_messages:
            content = str(msg.get("content", ""))

            match_stripped = re.search(
                r"<guess>\s*(.*?)\s*</guess>", content, re.DOTALL
            )
            match_raw = re.search(r"<guess>(.*?)</guess>", content, re.DOTALL)

            has_field = match_stripped is not None and match_stripped.group(1).strip()
            has_spacing = True
            if has_field and not (match_raw and match_raw.group(1)):
                has_spacing = False

            score = 0.0
            if has_field:
                score += 0.4
            if has_spacing:
                score += 0.2
            if content.strip().startswith("<guess>"):
                score += 0.2
            if content.strip().endswith("</guess>"):
                score += 0.2
            scores.append(score)

        return sum(scores) / len(scores) if scores else 0.0

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    # -- rubric wrappers (take state, delegate to existing helpers) --------

    def _reward_correct(self, state: RolloutState) -> tuple[float, str]:
        answer = state.sample.answer or ""
        messages = self._get_all_messages(state)
        return self._reward_correct_answer(messages, answer), answer

    def _reward_partial(self, state: RolloutState) -> float:
        answer = state.sample.answer or ""
        return self._reward_partial_answer(self._get_all_messages(state), answer)

    def _reward_length(self, state: RolloutState) -> float:
        answer = state.sample.answer or ""
        return self._reward_length_bonus(self._get_all_messages(state), answer)

    def _reward_fmt(self, state: RolloutState) -> float:
        return self._reward_format(self._get_all_messages(state))

    async def compute_reward(self, state: RolloutState, eos_token: str = ""):
        return await self.rubric.score(state=state)

