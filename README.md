# altico-transcribe

## Merry Christmas!

When we celebrated Christmas in Prague a couple years ago you told me about how frequently you attend dance performances and how diligently you've taken notes. I think I've figured out the basic pipeline for digitizing your notebooks so they can be tagged, searched, organized, analyzed, etc.

Try it out on the demo image (don't peek!) and see for yourself!

## Installation

1. Clone this repository and navigate to the project directory.
2. Install the required Python packages:

```sh
pip install -r requirements.txt
```

3. Set up your environment variables. Create a `.env` file in the project root with the following keys (I will give you these):

```
AZURE_VISION_ENDPOINT=your_azure_endpoint
AZURE_VISION_KEY=your_azure_key
OPENAI_API_KEY=your_openai_key
```

## Usage

To transcribe a handwritten notebook page image:

```sh
python demo.py path/to/image.jpg
```

**Options:**

- `--outdir DIR`   Output directory (default: `out`)
- `--no-clean`     Skip OpenAI cleanup step

Example:

```sh
python demo.py demo_image.jpeg
```

