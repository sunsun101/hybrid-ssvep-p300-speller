# AIT-thesis-template
This repository contains a latex template for writing your thesis in Asian Institute of Technology.

## Usage

### Uploading to Overleaf

To open this with overleaf, simply zip the entire file and upload it to overleaf.

```console
git clone https://github.com/AIT-brainlab/AIT-thesis-template.git # copies the repository to your computer
zip -r thesis_name.zip AIT-thesis-template # zips the repository to thesis_name.zip
```

> if you are subscribed to the premium overleaf package, simply fork this repo and sync it with your forked repo

### Locally compiling

If you have latex installed locally, you can use the following command to compile your thesis. The output pdf file will be named thesis.pdf.

```console
pdflatex thesis.tex && bibtex thesis.aux && pdflatex thesis.tex && pdflatex thesis.tex
```

> You can view the example output [here](https://github.com/ruke1ire/AIT-thesis-template/blob/main/thesis.pdf).

### How to use the template?

You can use the template by entering the information to the correct files.
- cover.tex contains the title, name, etc.
- declaration.tex contains the author's declaration
- contents.tex\* is the setup for the table of contents
- acknowledgements.tex is where you put your acknowledgements
- abstract.tex is where you write the abstract of your thesis
- loft.tex\* is the setup file for the list of figures and tables
- introduction.tex is where you write the introduction
- literature.tex is for your literature review
- methodology.tex is where you write your approach
- results.tex is where you write the results of your research
- conclusion.tex is where you give the summary and final remarks such as future work
- references.bib is where you enter the bibtex for each citations you make
- appendix.tex is where you enter your appendix
- thesis.tex is the main file which imports all the above files (you can comment out sections you do not need)

> You will probably not have to touch files containing \*.

## Contributing

Submit a pull request or open an issue if you find any bugs or would like to improve the template.

- Most of the setup such as the font size, line spacing, font style, etc. should be specified in aitthesis.cls.
- loft.tex and contents.tex contains the setup for the lists of figures, tables, and contents
- Add more files if necessary (be sure to document it in this README.md file)
