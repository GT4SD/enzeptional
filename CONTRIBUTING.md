# Contributing

## Contribution Terms and License

The documentation and benchmarking of GT4SD is contained in this repository. To contribute
to this project or any of the elements of GT4SD we recommend you start by reading this
contributing guide.

## Contributor License Agreement

Before you can submit any code we need all contributors to sign a
contributor license agreement. By signing a contributor license
agreement (CLA) you're basically just attesting to the fact
that you are the author of the contribution and that you're freely
contributing it under the terms of the MIT License.

When you contribute to the GT4SD project with a new pull request,
a bot will evaluate whether you have signed the CLA. If required, the
bot will comment on the pull request, including a link to accept the
agreement. The [individual CLA](./iCLA.md) document is available for review in this repo.
You can sign the agreement right away by opening an issue clicking [here](https://github.com/GT4SD/enzeptional/issues/new?assignees=&labels=cla-signing&template=cla-signature.yaml&title=CLA+signature).

## Contributing to ENZEPTIONAL codebase

If you would like to contribute to the package, we recommend the following development setup.

1. Create a copy of the [repository](https://github.com/GT4SD/enzeptional) via the "_Fork_" button.

2. Clone the enzeptional repository:

    ```sh
    git clone git@github.com:${GH_ACCOUNT_OR_ORG}/enzeptional.git
    ```

3. Add remote enzeptional repo as an "upstream" in your local repo, so you can check/update remote changes.

   ```sh
   git remote add upstream git@github.com:GT4SD/enzeptional.git
   ```

4. Create a dedicated branch:

    ```sh
    cd enzeptional
    git checkout -b a-super-nice-feature-we-all-need
    ```

5. Install `enzeptional` in editable mode:

    ```sh
    pip install -e.
    ```

6. Implement your changes and once you are ready run the tests:

    ```sh
    # this can take quite long a it's downloading all models and running them multiple times in the tests
    python -m pytest -sv
    ```

    And the style checks:

    ```sh
    # blacking and sorting imports (this might change your files)
    python -m black src/enzeptional
    python -m isort src/enzeptional
    # checking flake8 and mypy
    python -m flake8 --disable-noqa --per-file-ignores="__init__.py:F401" src/enzeptional
    python -m mypy src/enzeptional
    ```

    Ensure the license headers:

    ```sh
    licenseheaders -y 2024 -d src/enzeptional -o "GT4SD team" -t mit.tmpl
    ```

7. Once the tests and checks passes, but most importantly you are happy with the implemented feature, commit your changes.

    ```sh
    # add the changes
    git add 
    # commit them
    git commit -s -m "feat: implementing super nice feature." -m "A feature we all need."
    # check upstream changes
    git fetch upstream
    git rebase upstream/main
    # push changes to your fork
    git push -u origin a-super-nice-feature-we-all-need
    ```

8. From your fork, open a pull request via the "_Contribute_" button, the maintainers will be happy to review it.


