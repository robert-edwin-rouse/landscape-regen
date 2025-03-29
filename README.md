# landscape-regen

<a name="readme-top"></a>


<!-- PROJECT SHIELDS -->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]


<!-- PROJECT HEADER -->

[![Product Name Screen Shot][product-screenshot]](https://example.com)

<br />
<div align="center">
    <a href="https://github.com/robert-edwin-rouse/landscape-regen/issues">Report Bug</a>
    ·
    <a href="https://github.com/robert-edwin-rouse/landscape-regen/issues">Request Feature</a>
</div>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Prerequisites

The project requires Python 3.8+, and various libraries described in the
[requirements.txt](https://github.com/robert-edwin-rouse/landscape-regen/blob/main/requirements.txt)
file.

### Installation and Running

1. Clone the repository:
```sh
git clone https://github.com/robert-edwin-rouse/landscape-regen.git
cd landscape-regen
```

2. Update the submodules:
```sh
git submodule update --init --recursive
```

3. Install the dependencies. We recommend doing this from within a virtual environment, e.g.
```sh
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
(If you do not wish to use a virtual environment then just the last step can be used to install the dependencies globally).

4. Launch the dashboard:
```sh
python3 dashboard.py
```
This will notify you of the web address and port it is running on, e.g,

```
Dash is running on http://127.0.0.1:8051/
```
 
which you can now visit in your browser.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CONTRIBUTING -->
## Contributing

Contributions are welcome.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* This project received support from Schmidt Sciences, LLC via the [Institute of Computing for Climate Science](https://iccs.cam.ac.uk/)
* The UK hexagonal grid map data is from [ONSVisual](https://github.com/ONSvisual/topojson_boundaries).

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/robert-edwin-rouse/landscape-regen.svg?style=for-the-badge
[contributors-url]: https://github.com/robert-edwin-rouse/landscape-regen/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/robert-edwin-rouse/landscape-regen.svg?style=for-the-badge
[forks-url]: https://github.com/robert-edwin-rouse/landscape-regen/network/members
[stars-shield]: https://img.shields.io/github/stars/robert-edwin-rouse/landscape-regen.svg?style=for-the-badge
[stars-url]: https://github.com/robert-edwin-rouse/landscape-regen/stargazers
[issues-shield]: https://img.shields.io/github/issues/robert-edwin-rouse/landscape-regen.svg?style=for-the-badge
[issues-url]: https://github.com/robert-edwin-rouse/landscape-regen/issues
[license-shield]: https://img.shields.io/github/license/robert-edwin-rouse/landscape-regen.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[product-screenshot]: assets/Banner.png