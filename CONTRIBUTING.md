# Contributing to rssn-advanced

First of all, thank you for considering contributing to **rssn-advanced**! We are thrilled to have you. This project aims to become a next-generation scientific computing ecosystem in Rust, and every contribution, no matter how small, helps us get there.

We are building a friendly, open community and are committed to helping you get started.

---

## How Can I Contribute?

**We welcome contributions across all areas of the project!** Whether you are a seasoned Rust developer, a numerical methods expert, a student learning symbolic math, or just someone who wants to fix a typo, there is a place for you here.

If you have an idea, open an issue and we would be happy to discuss it!

---

## Getting Started: It's Easy!

We have designed the project to be as easy to set up as possible. You do not need any complex dependencies or special environment configuration.

1.  **Install Rust**: If you don't have it already, install the Rust toolchain via [rustup](https://rustup.rs/).

2.  **Fork and Clone**: Fork the repository on GitHub and clone it to your local machine.
    ```bash
    git clone https://github.com/YOUR-USERNAME/rssn-advanced.git
    cd rssn-advanced
    ```

3.  **Build and Test**: Check that everything is working correctly by running the test suite.
    ```bash
    cargo test --all
    ```

That's it! You are now ready to start contributing.

---

## 🔧 Development Workflow

1. **Create a feature branch**:

   ```bash
   git checkout -b feature/my-new-feature
   ```

2. **Write your code**: Make your changes, and please add tests for any new functionality or bug fixes!

3. **Code style & formatting**:

   * All code must compile with **zero warnings** on the latest stable Rust.
   * Additional **lint rules** are configured in `lib.rs` and must be respected.
   * Before creating a pull request, please always run:

     ```bash
     cargo +nightly fmt --all
     cargo clippy --all-targets -- -D warnings
     cargo test --all
     ```

4. **Commit and Push**:

   Commit your changes with a clear message and push them to your fork.

   ```bash
   git commit -m "feat(symbolic): add new integration method"
   ```

5. **Open a Pull Request**: Open a PR against the `main` branch of the `Apich-Organization/rssn-advanced` repository. Please provide a clear description of your changes.

## 🙏 Acknowledgements

Contributors are credited in release notes and on the GitHub page.
We value every contribution, from fixing typos to implementing new solvers.

Thank you for making **rssn-advanced** better!
