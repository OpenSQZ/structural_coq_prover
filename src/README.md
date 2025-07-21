# Modified Coq Source Files

This directory contains modified OCaml source files from Coq 8.11 that enable structured information extraction during compilation and proof checking.

## Overview

These modifications allow the Coq system to output detailed structural information including:
- All declarations and definitions in Coq files
- Complete type information for definitions  
- Proof state transitions during verification
- AST structure for theorem proving applications

## Important Note

**This directory contains only the modified files, not a complete Coq installation.** Due to the complexity of distributing a full modified Coq build, we provide only the changed source files that need to be applied to a clean Coq 8.11 installation.

## Installation Process

### Prerequisites
- Git
- OCaml compiler (compatible with Coq 8.11)
- Standard build tools (make, etc.)

### OCaml Installation

Coq 8.11 requires OCaml version 4.07.0 to 4.10.0. We **strongly recommend using OPAM** (OCaml Package Manager) to manage OCaml versions and dependencies, as it ensures proper version compatibility and dependency resolution.

#### Installing OPAM and OCaml

**On Ubuntu/Debian:**
```bash
# Install OPAM
sudo apt-get update
sudo apt-get install opam build-essential

# Initialize OPAM (creates ~/.opam directory)
# For Docker environments:
opam init --disable-sandboxing -y
# For regular systems:
# opam init -y

eval $(opam env)

# Create and switch to OCaml 4.10.0 environment
opam switch create coq-8.11 4.10.0
eval $(opam env)

# Verify OCaml installation
ocaml -version
# Should output: The OCaml toplevel, version 4.10.0
```

**On macOS:**
```bash
# Install OPAM via Homebrew
brew install opam

# Initialize OPAM
# For Docker environments:
opam init --disable-sandboxing -y
# For regular systems:
# opam init -y

eval $(opam env)
opam switch create coq-8.11 4.10.0
eval $(opam env)
```

#### Installing Coq Dependencies

After setting up OCaml, install the required build dependencies:

```bash
# Essential build tools
opam install dune num ocamlfind

# Additional dependencies for Coq 8.11
opam install camlp5
```

**Important Notes:**
- Always run `eval $(opam env)` after switching OPAM environments
- The `coq-8.11` switch name is recommended to avoid conflicts with other projects
- **Docker users**: Must use `--disable-sandboxing` flag when initializing OPAM, as Docker containers don't support OPAM's sandboxing features
- **Regular systems**: Use `opam init -y` without the disable-sandboxing flag for better security
- If you encounter permission issues, ensure OPAM was initialized properly

### Step-by-Step Installation

1. **Clone Coq 8.11 source:**
   ```bash
   git clone --branch V8.11.2 https://github.com/coq/coq.git coq-8.11
   cd coq-8.11
   ```

2. **Apply our patches using the provided script:**
   ```bash
   # From the src/ directory
   ./patch.sh coq-8.11 /path/to/install
   ```
   
   This script will:
   - Copy our modified files to the correct locations in the Coq source tree
   - Display the exact build commands you need to run
   - Show the final binary locations after installation
   
   **Follow the script's output carefully** - it provides complete build instructions and tells you where the binaries will be installed.

3. **Manual patching (alternative):**
   If you prefer manual control over file replacement:
   ```bash
   # Copy files to their correct locations in Coq source
   # (see patch.sh for complete file mappings)
   cp src/names.ml coq-8.11/kernel/names.ml
   cp src/names.mli coq-8.11/kernel/names.mli
   cp src/pretyping.ml coq-8.11/pretyping/pretyping.ml
   cp src/typeops.ml coq-8.11/kernel/typeops.ml
   # ... and so on for all modified files
   
   # Then build as above
   cd coq-8.11
   ./configure -prefix /path/to/install
   make -j $(nproc)
   make install
   ```

### Using the Modified Coq

After installation, use the modified Coq compiler:

```bash
export PATH=/path/to/install/bin:$PATH
coqc your_file.v  # Will output structured information
```

The modified compiler will output additional structured data that can be consumed by the theorem proving framework.

**Note:** Using coqc directly requires complex dependencies and build setup. For easier usage, we provide Python implementations in the `/root/coqc.py` that wrap the modified Coq functionality and are more convenient to call from machine learning workflows.

### Library Installation

**Standard Coq Libraries:** Normal Coq (distributed through OCaml/opam) can directly install existing libraries through the OCaml package manager. Our modified Coq can also be linked with OCaml to use these libraries, but **this is not recommended** due to potential compatibility issues.

**Recommended Approach:** We handle library dependencies by cloning source code directly and building from source, which provides better control over the compilation process with our modifications.

### Known Issues and Workarounds

Due to our kernel-level modifications, some compilation issues may occur. This is one of the main reasons we plan to transition to a plugin architecture in the future.

**Common Problem:** Some theorems may fail to compile due to our structural analysis modifications.

**Solution:** Use `Unset Linear Execution.` before problematic theorems and `Set Linear Execution.` after completion:

```coq
(* Before a problematic theorem *)
Unset Linear Execution.

Theorem problematic_theorem : forall n : nat, n + 0 = n.
Proof.
  intro n.
  induction n as [| n' IHn'].
  - reflexivity.
  - simpl. rewrite IHn'. reflexivity.
Qed.

(* Restore normal execution *)  
Set Linear Execution.

(* Continue with other definitions *)
Definition next_function := ...
```

This workaround disables certain internal optimizations that may conflict with our structural extraction, allowing the theorem to compile successfully.

## File Overview

The modified files span multiple components of Coq:

### Core Type System
- `typeops.ml` - Kernel type operations with structural extraction
- `pretyping.ml` - Pretyping phase modifications  
- `names.ml`, `names.mli` - Name handling with metadata

### Parsing and Printing
- `cLexer.ml` - Lexical analysis modifications
- `printer.ml` - Enhanced proof term printing
- `pp*.ml` - Pretty printing with structural info
- `detyping.ml` - Type inference reverse operations

### Vernacular Commands
- `com*.ml` - Command handling (definitions, inductives, etc.)
- `vernacentries.ml` - Vernacular command processing
- `vernacinterp.ml` - Vernacular interpretation

### Proof System
- `proof_global.ml` - Global proof state management
- `tacinterp.ml` - Tactic interpretation
- `tacentries.ml` - Tactic registration

### Libraries and Environment
- `globEnv.ml` - Global environment extensions
- `nametab.ml` - Name table modifications
- `constrintern.ml` - Term internalization

## Compatibility

These modifications are specifically designed for **Coq 8.11.2**. Using them with other Coq versions may require additional porting work due to API changes between versions.

## Integration with ML Framework

The structured information output by this modified Coq is consumed by the Python machine learning framework in the `data_extraction` directory. The integration points include:

- **Proof state extraction** for training neural tactic generators
- **Definition parsing** for building semantic embeddings
- **Type information** for context-aware proof search

For details on using the ML framework with this modified Coq, see the main project documentation.

## Future Development Plans

**Note:** The current approach is highly invasive, requiring direct modification of Coq's source code. This creates maintenance challenges and limits compatibility to a specific Coq version.

**Future Plan:** We plan to refactor this system into a **Coq plugin architecture**. This would provide several advantages:

- **Version Compatibility:** Support for multiple Coq versions without source modifications
- **Plug-and-Play:** Easy installation and usage without rebuilding Coq
- **Maintenance:** Reduced coupling with Coq internals
- **Distribution:** Simpler packaging and deployment

The plugin approach would use Coq's official plugin API to hook into the compilation process and extract the same structural information currently obtained through source modifications. This will make the tool much more accessible and maintainable for the broader Coq community.