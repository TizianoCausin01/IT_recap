# Code Generation Commandments

Read this file before generating or editing code in this project. Treat it as the default coding contract unless the user explicitly asks for something different.

1. Prefer interpretability over cleverness.
   Write code that is easy to read, debug, and modify. Avoid overly abstract, overly generic, or "too smart" solutions unless they clearly reduce real complexity.

2. Match the existing code style first.
   Inspect `src/` and `/Users/tizianocausin/Desktop/useful_stuff/python_scripts/src/useful_stuff` before introducing patterns. Keep naming, imports, file organization, save-name conventions, and scientific Python style consistent with the existing code.

3. Keep code brief when possible.
   Do not add unnecessary boilerplate. Prefer compact, focused functions with direct control flow. Split code only when it improves readability or reuse.

4. Reuse existing utilities.
   Before writing a helper, check whether `useful_stuff` or the project `src/` already provides it. Prefer clear existing utilities such as `TimeSeries`, `print_wise`, RDM/RSA/II helpers, and project-specific loaders when appropriate.

5. Soft-code paths and environment values.
   Avoid hard-coded local paths, environment names, and configuration values when they can come from `config.yaml`, `MY_ENV`, or `Path(__file__)`.

```python
ENV = os.getenv("MY_ENV", "tiziano_mac_mini")
PROJECT_ROOT = Path(__file__).resolve().parents[3]

with open(PROJECT_ROOT / "config.yaml", "r") as f:
    config = yaml.safe_load(f)

paths = config[ENV]["paths"]
sys.path.append(paths["src_path"])
sys.path.append(paths["useful_stuff_path"])
```

6. Use the established function documentation style.
   Every new non-trivial function should use this structure, preferably before the function definition when matching nearby files:

```python
"""
function_name
Short explanation of what the function does.

INPUT:
    - arg_name: type -> explanation
    - another_arg: type -> explanation

OUTPUT:
    - output_name: type -> explanation
"""
def function_name(...):
    ...
# EOF
```

7. Add explicit end-of-block comments where they help.
   Use concise comments such as `# end if ...`, `# end for ...`, `# end while ...`, `# end try`, `# EOF`, and `# EOC`, especially for nested or long blocks. Do not add them where they become noise.

8. Be modular, not over-engineered.
   Prefer small functions over large tangled blocks, but avoid unnecessary classes, factories, decorators, or framework-like abstractions.

9. Preserve the project structure.
   Use imports, paths, and module placement that fit the current repository. Do not change folder structure unless explicitly asked.

10. Make generated code easy to inspect.
    Use clear variable names, readable loops, and simple conditionals. Avoid dense one-liners when they hide intent. Add comments for non-obvious logic, assumptions, and scientific/data-shape decisions.

11. Be conservative with dependencies.
    Do not introduce new packages unless necessary. Prefer the standard library and dependencies already used in this project or `useful_stuff`.

12. Surface conflicts with these rules.
    If a requested implementation conflicts with this file, mention the conflict and choose the most practical solution for the task.

13. Use a config object to specify the parameters of my code.
    If some arguments have to be passed, use argparse in the same way.

For example:

```python
@dataclass
class Cfg:
    folder_name: str = "talia_20each_tizi"
    model_name: str = "vit_l_16"
    img_size: int = 384
    k: int = 1
    metric_A: str = "cosine_cnt"
```

# EOF
