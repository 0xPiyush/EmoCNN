<script lang="ts">
	import cv from "@techstark/opencv-js";
	import { onMount } from "svelte";
	import cascade from "./haarcascade_frontalface_default.xml?raw";

	import * as ort from "onnxruntime-web";

	let canvas: HTMLCanvasElement;
	let classifier: cv.CascadeClassifier;
	let ortSession: ort.InferenceSession;

	const labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"];

	let expression = "Neutral";

	// Display webcam feed on canvas
	async function displayVideo() {
		const video = document.createElement("video");
		const stream = await navigator.mediaDevices.getUserMedia({
			video: {
				width: 320,
				height: 240
			}
		});

		video.addEventListener("loadedmetadata", () => {
			video.width = video.videoWidth;
			video.height = video.videoHeight;

			let src = new cv.Mat(video.height, video.width, cv.CV_8UC4);
			let dst = new cv.Mat(video.height, video.width, cv.CV_8UC4);
			let gray = new cv.Mat();
			let cap = new cv.VideoCapture(video);
			let faces = new cv.RectVector();

			const drawFrame = () => {
				cap.read(src);
				src.copyTo(dst);
				cv.cvtColor(dst, gray, cv.COLOR_RGBA2GRAY, 0);

				// detect faces.
				classifier.detectMultiScale(gray, faces, 1.1, 3, 0);
				// draw faces.
				for (let i = 0; i < faces.size(); ++i) {
					let face = faces.get(i);
					let point1 = new cv.Point(face.x, face.y);
					let point2 = new cv.Point(face.x + face.width, face.y + face.height);
					cv.rectangle(dst, point1, point2, [255, 0, 0, 255]);
				}

				// Get face image
				if (faces.size() > 0) {
					let face = faces.get(0);
					let faceMat = gray.roi(face);
					// Resize face image to 48x48
					cv.resize(faceMat, faceMat, new cv.Size(48, 48), 0, 0, cv.INTER_AREA);
					runModel(faceMat);
					faceMat.delete();
				}

				cv.imshow(canvas, dst);

				requestAnimationFrame(drawFrame);
			};
			requestAnimationFrame(drawFrame);
		});
		video.srcObject = stream;
		video.play();
	}

	function loadCascadeClassifier() {
		// Convert cascade to Uint8Array
		const data = new Uint8Array(cascade.length);
		for (let i = 0; i < cascade.length; i++) {
			data[i] = cascade.charCodeAt(i);
		}
		cv.FS_createDataFile("/", "haarcascade_frontalface_default.xml", data, true, false, false);

		// Load cascade classifier
		classifier = new cv.CascadeClassifier();
		classifier.load("haarcascade_frontalface_default.xml");
		console.log("Cascade classifier loaded");
	}

	async function loadOnnxModel() {
		ortSession = await ort.InferenceSession.create("/model.onnx", {
			executionProviders: ["webgl"],
			graphOptimizationLevel: "all"
		});
		console.log("ONNX model loaded");
	}

	function indexOfMax(arr: Float32Array) {
		let maxIndex = 0;
		for (let i = 1; i < arr.length; i++) {
			if (arr[i] > arr[maxIndex]) {
				maxIndex = i;
			}
		}
		return maxIndex;
	}

	async function runModel(img: cv.Mat) {
		const input = new ort.Tensor(Float32Array.from(img.data));
		const feeds = { conv2d_input: input };
		const output = await ortSession.run(feeds);
		let result = output[ortSession.outputNames[0]].data;
		console.log(result);
		let maxIndex = indexOfMax(result as Float32Array);
		expression = labels[maxIndex];
	}

	onMount(async () => {
		cv.onRuntimeInitialized = () => {
			loadCascadeClassifier();
			displayVideo();
		};
		await loadOnnxModel();
	});
</script>

<canvas bind:this={canvas} />
<h1>Expression: {expression}</h1>

<style>
	/* canvas {
		width: 50%;
		height: 50%;
	} */
</style>
