import argparse
import os
import onnx


def get_model_info(onnx_path):
    """Extract input/output tensor names and shapes from an ONNX model."""
    model = onnx.load(onnx_path)

    # Get input info
    input_info = model.graph.input[0]
    input_name = input_info.name
    input_shape = [dim.dim_value for dim in input_info.type.tensor_type.shape.dim]

    # Get output info
    output_info = model.graph.output[0]
    output_name = output_info.name

    return {
        "input_name": input_name,
        "input_shape": input_shape,
        "output_name": output_name,
    }


def generate_html(model_path, model_info, title="Digit Recognition"):
    """Generate HTML content for the digit recognition demo."""
    input_shape = model_info["input_shape"]
    input_name = model_info["input_name"]
    output_name = model_info["output_name"]

    # Determine tensor shape string based on model type
    # MLP: [1, 28, 28], CNN: [1, 1, 28, 28]
    if len(input_shape) == 3:
        tensor_shape = "[1, 28, 28]"
    else:
        tensor_shape = "[1, 1, 28, 28]"

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: monospace;
            max-width: 600px;
            margin: 2rem auto;
            padding: 1rem;
        }}

        #drawCanvas {{
            border: 1px solid #000;
            cursor: crosshair;
            image-rendering: pixelated;
            width: 280px;
            height: 280px;
        }}

        .buttons {{
            margin: 1rem 0;
        }}

        button {{
            font-family: monospace;
            padding: 0.5rem 1rem;
            cursor: pointer;
        }}

        #recognizeBtn:disabled {{
            opacity: 0.5;
            cursor: not-allowed;
        }}

        .result {{
            margin: 1.5rem 0;
            padding: 1rem;
            border: 1px solid #ccc;
        }}

        .prediction-value {{
            font-size: 3rem;
            font-weight: bold;
        }}

        .probabilities {{
            margin-top: 1rem;
            font-size: 0.875rem;
        }}

        .prob-row {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin: 0.25rem 0;
        }}

        .prob-row .digit {{
            width: 1rem;
        }}

        .prob-row .bar-bg {{
            flex: 1;
            height: 0.75rem;
            background: #eee;
        }}

        .prob-row .bar {{
            height: 100%;
            background: #666;
        }}

        .prob-row .bar.top {{
            background: #000;
        }}

        .prob-row .pct {{
            width: 3rem;
            text-align: right;
        }}
    </style>
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>
</head>
<body>
    <h1>{title}</h1>
    <p>Draw a digit (0-9) and click "Recognize".</p>

    <div>
        <canvas id="drawCanvas" width="28" height="28"></canvas>
        <div style="font-size: 0.75rem; color: #666;">28x28 pixels (scaled 10x for display)</div>
    </div>

    <div class="buttons">
        <button id="recognizeBtn" disabled>Recognize</button>
        <button id="clearBtn">Clear</button>
    </div>

    <div class="result">
        <div>Prediction: <span class="prediction-value" id="prediction">?</span></div>
        <div id="confidence"></div>

        <div class="probabilities" id="probabilities"></div>
    </div>

    <script>
        let session = null;

        function softmax(x) {{
            const maxVal = Math.max(...x);
            const exps = x.map(v => Math.exp(v - maxVal));
            const sum = exps.reduce((a, b) => a + b, 0);
            return exps.map(v => v / sum);
        }}

        async function runInference(inputData) {{
            const tensor = new ort.Tensor('float32', inputData, {tensor_shape});
            const results = await session.run({{ {input_name}: tensor }});
            return Array.from(results.{output_name}.data);
        }}

        const canvas = document.getElementById('drawCanvas');
        const ctx = canvas.getContext('2d', {{ willReadFrequently: true }});

        let isDrawing = false;
        let lastX = 0;
        let lastY = 0;

        function clearCanvas() {{
            ctx.fillStyle = '#000';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            document.getElementById('prediction').textContent = '?';
            document.getElementById('confidence').textContent = '';
            updateProbabilityBars([]);
        }}

        clearCanvas();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';

        function getPos(e) {{
            const rect = canvas.getBoundingClientRect();
            const scaleX = canvas.width / rect.width;
            const scaleY = canvas.height / rect.height;

            if (e.touches) {{
                return {{
                    x: (e.touches[0].clientX - rect.left) * scaleX,
                    y: (e.touches[0].clientY - rect.top) * scaleY
                }};
            }}
            return {{
                x: (e.clientX - rect.left) * scaleX,
                y: (e.clientY - rect.top) * scaleY
            }};
        }}

        function startDrawing(e) {{
            isDrawing = true;
            const pos = getPos(e);
            lastX = pos.x;
            lastY = pos.y;
            ctx.beginPath();
            ctx.arc(lastX, lastY, ctx.lineWidth / 2, 0, Math.PI * 2);
            ctx.fillStyle = '#fff';
            ctx.fill();
        }}

        function draw(e) {{
            if (!isDrawing) return;
            e.preventDefault();

            const pos = getPos(e);
            ctx.beginPath();
            ctx.moveTo(lastX, lastY);
            ctx.lineTo(pos.x, pos.y);
            ctx.stroke();

            lastX = pos.x;
            lastY = pos.y;
        }}

        function stopDrawing() {{
            isDrawing = false;
        }}

        canvas.addEventListener('mousedown', startDrawing);
        canvas.addEventListener('mousemove', draw);
        canvas.addEventListener('mouseup', stopDrawing);
        canvas.addEventListener('mouseout', stopDrawing);
        canvas.addEventListener('touchstart', startDrawing);
        canvas.addEventListener('touchmove', draw);
        canvas.addEventListener('touchend', stopDrawing);

        function getImageData() {{
            const imageData = ctx.getImageData(0, 0, 28, 28);
            const pixels = imageData.data;

            const input = new Float32Array(784);
            for (let i = 0; i < 784; i++) {{
                input[i] = pixels[i * 4] / 255.0;
            }}

            return input;
        }}

        function updateProbabilityBars(probs) {{
            const container = document.getElementById('probabilities');

            if (probs.length === 0) {{
                container.innerHTML = Array.from({{length: 10}}, (_, i) => `
                    <div class="prob-row">
                        <span class="digit">${{i}}</span>
                        <div class="bar-bg"><div class="bar" style="width: 0%"></div></div>
                        <span class="pct">-</span>
                    </div>
                `).join('');
                return;
            }}

            const maxIdx = probs.indexOf(Math.max(...probs));

            container.innerHTML = probs.map((p, i) => `
                <div class="prob-row">
                    <span class="digit">${{i}}</span>
                    <div class="bar-bg"><div class="bar ${{i === maxIdx ? 'top' : ''}}" style="width: ${{(p * 100).toFixed(1)}}%"></div></div>
                    <span class="pct">${{(p * 100).toFixed(1)}}%</span>
                </div>
            `).join('');
        }}

        async function recognize() {{
            if (!session) {{
                alert('Model not loaded yet!');
                return;
            }}

            const input = getImageData();
            const logits = await runInference(input);
            const probs = softmax(logits);

            let maxIdx = 0;
            let maxVal = probs[0];
            for (let i = 1; i < probs.length; i++) {{
                if (probs[i] > maxVal) {{
                    maxVal = probs[i];
                    maxIdx = i;
                }}
            }}

            document.getElementById('prediction').textContent = maxIdx;
            document.getElementById('confidence').textContent =
                `Confidence: ${{(maxVal * 100).toFixed(1)}}%`;

            updateProbabilityBars(probs);
        }}

        document.getElementById('recognizeBtn').addEventListener('click', recognize);
        document.getElementById('clearBtn').addEventListener('click', clearCanvas);

        updateProbabilityBars([]);

        ort.InferenceSession.create('{model_path}')
            .then(s => {{
                session = s;
                document.getElementById('recognizeBtn').disabled = false;
            }})
            .catch(err => {{
                console.error('Failed to load model:', err);
                alert('Failed to load model. See console for details.');
            }});
    </script>
</body>
</html>'''

    return html


def main():
    parser = argparse.ArgumentParser(
        description="Generate an HTML demo page from an ONNX digit recognition model"
    )
    parser.add_argument(
        "onnx_file",
        type=str,
        help="Path to the ONNX model file",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output HTML file path (default: same name as input with .html extension)",
    )
    parser.add_argument(
        "-t", "--title",
        type=str,
        default="Digit Recognition",
        help="Page title (default: 'Digit Recognition')",
    )
    parser.add_argument(
        "-m", "--model-path",
        type=str,
        help="Path to model in the HTML (default: same as input file basename)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.onnx_file):
        print(f"Error: ONNX file not found: {args.onnx_file}")
        return 1

    # Get model info
    model_info = get_model_info(args.onnx_file)
    print(f"Model input: {model_info['input_name']} {model_info['input_shape']}")
    print(f"Model output: {model_info['output_name']}")

    # Determine output path
    output_path = args.output
    if not output_path:
        model_name = os.path.splitext(os.path.basename(args.onnx_file))[0]
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(script_dir, "visualizations", "demo")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{model_name}.html")

    # Determine model path for HTML
    model_path = args.model_path
    if not model_path:
        model_path = os.path.abspath(args.onnx_file)

    # Generate and write HTML
    html = generate_html(model_path, model_info, args.title)
    with open(output_path, "w") as f:
        f.write(html)

    print(f"Generated: {output_path}")
    return 0


if __name__ == "__main__":
    exit(main())
