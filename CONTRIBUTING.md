# Contributing to SSL in the Wild project

If you are not sure about something, feel free to ask in the discussion of the issue. We are happy to help!

## How to contribute

0. Check for already existing issues if you have a feature request.
1. **Create new issue** for any bugs or feature requests.
2. **Fork** the repository.
3. **Create a new branch** from `main` for each issue.
4. Make your changes.
5. Pull in the `main` branch into your branch and resolve any merge conflicts.
6. Open a **pull request** into `main` branch.
7. Link the pull request by mentioning the **issue number** `#<issue_number>` in the description or any other way.
8. Set reviewer on pull request (or mention people in comments).
9. Discuss any problems or suggestions.
10. Make changes if necessary.
11. **Congratulations!** You have contributed to the project!

## Conventions

To keep the code clean and readable, we are using coding conventions and rules. Make sure to follow them where possible.

The project is mainly written in Python and follows the [PEP 8 style guide](https://www.python.org/dev/peps/pep-0008/). Also use static typing where possible.

## Documentation

Document your code with comments where necessary, follow coding convention rules.

### Feature Documentation

For each larger feature, like implementing `wandb` support, make sure to write a proper, but not too extensive documentation using `Markdown`. It should be understandable for someone who is not familiar with the project. For this smaller project, we recommend you to write the documentation in the `README.md` file. Documentation style is based on [Make a README](https://www.makeareadme.com/) template.

For `Markdown` documentation, use a linter to check for formatting errors. We strongly recommend [markdownlint](https://github.com/DavidAnson/markdownlint) to help with consistency. VSCode extension available [here](https://marketplace.visualstudio.com/items?itemName=DavidAnson.vscode-markdownlint).

If you see a document referencing related features or issues, make sure to **update them**. Like make a change in the main `README.md` to list new features you added.

### Diagrams, Sketches and Assets

Including diagrams/sketches and assets is not necessary, but can be helpful to understand the feature.

For new features or examples, you can create diagrams or include pictures in the documentation, saved in the locally created `assets` folder.

If you want to include diagrams or sketches, you can use [Excalidraw](https://excalidraw.com/) and export them as `SVG` files to the locally created `assets` folder. You can save the `.excalidraw` files in the `assets` folder to modify later if needed. You can also use other tools.
