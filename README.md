# LLM-based extension of the Financial Support Network (FSN) model of Fuentes, Schmutzler and Carreón-Vazquez (2026)

## Scope

This package contains the code for the LLM-based investor-selection extension developed for the FSN agent-based model. It does **not** contain the complete original NetLogo model. The underlying FSN model is the model described in Fuentes et al. (2026); users must obtain and use that original model separately and in accordance with its applicable license and access conditions.

## Package contents

- `llm_vc_extension.txt` — the NetLogo procedures associated with the LLM extension. It includes the modified `mating-vc` dispatcher and the procedures `initialize-vc-python`, `llm-vc-selection`, and `make-vc-candidate-packet`.
- `vc_brain.py` — the Python module that formats candidate information, maintains the investor's episodic memory, queries Ollama, validates the selected startup ID, and returns the decision to NetLogo.

## Requirements

- NetLogo 7.0.4.
- The Python extension for NetLogo (`py`) installed and configured for the Python interpreter used by NetLogo.
- Python and the `ollama` package available to that same interpreter.
- Ollama installed and running locally.
- The model `llama3.2:3b` downloaded in Ollama.
- The original FSN model, obtained separately.

Install the Python package in the environment used by NetLogo:

```bash
python -m pip install ollama
```

Download the model:

```bash
ollama pull llama3.2:3b
```

## Integration with the original NetLogo model

1. Make a working copy of the original FSN model. Do not overwrite the original source file.
2. Keep `vc_brain.py` in the Python working directory from which NetLogo imports Python modules. If necessary, add that directory to Python's module search path.
3. Open the working copy in NetLogo 7.0.4.
4. Confirm that the model already declares the `py` extension. The original model's `extensions` declaration should include `py`.
5. Add the procedures from `llm_vc_extension.txt` to the Code tab.
6. Replace the original `mating-vc` procedure with the version included in `llm_vc_extension.txt`. The other three procedures are additional procedures.
7. Confirm that the model's global variables include `python-ready?` and `selected-by-vc`, and that its startup/turtle variables used by the extension are present. These are inherited dependencies from the original FSN model, not newly defined by this package.
8. Select the model's `LLM` option for `vc-selection` and run `setup`. The Python bridge is initialized when the LLM selection mechanism is first invoked; investor memory is reset at initialization and retained across subsequent funding decisions in that simulation run.

## Model variables required by the extension

The extension expects the original FSN model to define the following global variables:

- `python-ready?`
- `selected-by-vc`
- `#male2`, `#female2`, `#mix2`, `#total`
- `come-acc0`, `come-acc1`, `come-acc2`
- `not-come-acc0`, `not-come-acc1`, `not-come-acc2`

It also expects the original startup/turtle variables and reporters used in the procedures, including `want-vc`, `meeting`, `mating`, `d`, `gender`, `selection2?`, `my-alters_m`, `my-alters_f`, `my-alters_x`, `social-net`, `tie-vc`, `funds`, `initial-state`, `accelerated`, `mating-acc`, `accelerated-period`, `selection1?`, `meet_prob`, `mat_prob`, and `want-acc`, as well as the standard NetLogo `who`, `ticks`, `link-neighbors`, and `py` extension primitives.

The candidate packet contains 25 fields per eligible startup. The field order is defined in `vc_brain.py` (`FIELD_ORDER`) and must remain synchronized with `make-vc-candidate-packet` in the NetLogo extension.

## Decision and memory behavior

At each VC decision, NetLogo identifies eligible applicants, updates their counts of successful connected startups, and sends their modeled attributes to Python. Python passes the candidate records and the investor's accumulated memory to Ollama. The LLM returns the ID of one currently eligible startup. Python validates that ID and stores the selected startup's characteristics at the time of investment in the investor's memory. The memory is reset at the beginning of a new simulation run.

The LLM mechanism does not impose a fixed numerical score, ranking formula, attribute weights, or gender quota. The code includes a one-candidate path and a corrective response attempt if the LLM returns an unavailable ID; if the response remains invalid, NetLogo uses its seeded random generator for a fallback selection and records that selection in memory.

## Reproducibility and attribution

This repository documents and distributes the LLM extension only. It does not redistribute the original FSN model or replace the citation to the original model. Please cite Fuentes et al. (2026) when using the underlying FSN model, and cite the article introducing this LLM extension when using the extension itself.

Before redistributing this extension or combining it with the original model, users should check the license and reuse conditions applicable to the original FSN model. The presence of a citation does not, by itself, grant permission to redistribute third-party or jointly authored source code.

## Validation status

The Python source was checked for syntax compilation when this package was prepared. This package has not been live-tested here with a running NetLogo 7.0.4, the NetLogo Python extension, and Ollama together. Users should verify the integration in their local environment before running experiments.
