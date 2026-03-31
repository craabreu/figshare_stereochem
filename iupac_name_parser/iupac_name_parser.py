import pathlib
import re

import jpype
import jpype.imports  # noqa: F401

OPSIN_JAR_PATH = (
    pathlib.Path(__file__).parent / "opsin-cli-2.9.0-jar-with-dependencies.jar"
)

if not jpype.isJVMStarted():
    jpype.startJVM("--enable-native-access=ALL-UNNAMED", classpath=[OPSIN_JAR_PATH])

from uk.ac.cam.ch.wwmm import opsin  # noqa: E402

LABEL_REGEX = re.compile(r"\s*\(?\d+[A-Za-z]*\)?\s*$")

OPENING_BRACKETS = "{[("
CLOSING_BRACKETS = ")]}"
BRACKET_PAIRS = {
    "(": ")",
    "[": "]",
    "{": "}",
    "}": "{",
    "]": "[",
    ")": "(",
}


def find_substring(
    string: str,
    substring: str,
    *,
    ignored: str = " ",
) -> tuple[int, int]:
    ignored = set(ignored)
    n, m = len(string), len(substring)

    for start in range(n):
        i, j = start, 0
        first = -1

        while True:
            while j < m and substring[j] in ignored:
                j += 1
            if j == m:
                return first, i
            while i < n and string[i] in ignored:
                i += 1
            if i >= n or string[i] != substring[j]:
                break
            if first == -1:
                first = i
            i += 1
            j += 1

    return -1, -1


class IUPACNameParser:
    def __init__(
        self,
        imply_acid: bool = False,
        allow_radicals: bool = False,
        use_wildcards: bool = False,
        ignore_bad_stereo: bool = False,
        analyze_failures: bool = False,
        fix_uninterpretable: bool = True,
        fix_brackets: bool = True,
        verbose: bool = False,
    ):
        self._nts = opsin.NameToStructure.getInstance()
        self._nti = opsin.NameToInchi()
        self._config = opsin.NameToStructureConfig()
        self._config.setInterpretAcidsWithoutTheWordAcid(imply_acid)
        self._config.setAllowRadicals(allow_radicals)
        self._config.setOutputRadicalsAsWildCardAtoms(use_wildcards)
        self._config.setWarnRatherThanFailOnUninterpretableStereochemistry(
            ignore_bad_stereo
        )
        self._config.setDetailedFailureAnalysis(analyze_failures)
        self._fix_uninterpretable = fix_uninterpretable
        self._fix_brackets = fix_brackets
        self._verbose = verbose

    @staticmethod
    def _preprocess_name(name: str) -> str:
        name = LABEL_REGEX.sub("", name)
        return re.sub(r"-\s+", "-", name).strip()

    @staticmethod
    def _fix_uninterpretable_molecule(name: str, error: str) -> str:
        section_start = error.find("name: ") + 6
        section_end = error.find("  The following", section_start)
        if section_end == -1:
            section_end = len(error)
        section = error[section_start:section_end]
        start, stop = find_substring(name, section)
        if start == -1:
            return ""
        return name[:start] + name[stop:]

    @staticmethod
    def _fix_unmatched_opening_bracket(name: str) -> str:
        closing_brackets = []
        for char in reversed(name):
            if char in OPENING_BRACKETS:
                pair = BRACKET_PAIRS[char]
                if closing_brackets and pair == closing_brackets[-1]:
                    closing_brackets.pop()
                else:
                    return name + pair
            elif char in CLOSING_BRACKETS:
                closing_brackets.append(char)
        return ""

    @staticmethod
    def _fix_unmatched_closing_bracket(name: str) -> str:
        opening_brackets = []
        for char in name:
            if char in CLOSING_BRACKETS:
                pair = BRACKET_PAIRS[char]
                if opening_brackets and pair == opening_brackets[-1]:
                    opening_brackets.pop()
                else:
                    return pair + name
            elif char in OPENING_BRACKETS:
                opening_brackets.append(char)
        return ""

    def _parse_name(
        self, name: str, original_name: str | None = None
    ) -> tuple[str, opsin.OpsinResult]:
        if original_name is None:
            original_name = name
        name = self._preprocess_name(name)
        if name == "":
            raise ValueError("Empty name")
        if self._verbose:
            print(f"Name: <{name}>")
        result = self._nts.parseChemicalName(name, self._config)
        if result.getStatus() == opsin.OpsinResult.OPSIN_RESULT_STATUS.FAILURE:
            error = str(result.getMessage())
            if self._fix_uninterpretable and "uninterpretable" in error:
                fixed = self._fix_uninterpretable_molecule(name, error)
                if fixed and fixed != name:
                    return self._parse_name(fixed, original_name)
            elif self._fix_brackets and error.startswith("Unmatched opening bracket"):
                fixed = self._fix_unmatched_opening_bracket(name)
                if fixed:
                    return self._parse_name(fixed, original_name)
            elif self._fix_brackets and error.startswith("Unmatched closing bracket"):
                fixed = self._fix_unmatched_closing_bracket(name)
                if fixed:
                    return self._parse_name(fixed, original_name)
            elif "unsure" in error:
                raise ValueError(f"Unable to parse '{original_name}'")
            raise ValueError(error)
        return name, result

    def to_smiles(self, name: str) -> tuple[str, str]:
        name, result = self._parse_name(name)
        return name, str(result.getSmiles())

    def to_cml(self, name: str) -> tuple[str, str]:
        name, result = self._parse_name(name)
        return name, str(result.getCml())

    def to_inchi(
        self, name: str, fixed_hydrogen_layer: bool = False
    ) -> tuple[str, str]:
        name, result = self._parse_name(name)
        if fixed_hydrogen_layer:
            return name, str(self._nti.convertResultToInChI(result))
        return name, str(self._nti.convertResultToStdInChI(result))

    def to_inchi_key(self, name: str) -> tuple[str, str]:
        name, result = self._parse_name(name)
        return name, str(self._nti.convertResultToStdInChIKey(result))
