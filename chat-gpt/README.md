# Northstar Vision

A private, browser-based object, face, and voice-cue recognition app. It labels common objects and recognizes people only after you enroll them with your own samples.

## Run it

1. Open a terminal in this folder.
2. Run `node server.mjs`.
3. Visit [http://localhost:4173](http://localhost:4173) in a modern browser.

The first launch needs an internet connection to obtain the public AI model files from the TensorFlow and face-api public model hosts. After the models are cached by the browser, they may be available without a connection depending on browser cache settings.

The live camera is tuned for ordinary laptops: it uses the browser's WebGL GPU path when available, checks faces about once per second, and refreshes object labels about every 2.4 seconds. This makes face labels respond quickly while avoiding overlapping, GPU-heavy recognition passes.

## How face memory works

When you add a name and one to six clear photos, the app finds the face in each photo and turns it into a 128-number face descriptor. It saves that descriptor and the label you supplied in the browser's local IndexedDB database; it does **not** retain the original enrollment images.

When a photo or camera frame is scanned, the app makes another descriptor and compares its mathematical distance against the local profiles. A close result (threshold `0.52`) gets the stored name. Otherwise it is labelled **Unknown face**. Enrolling a few photos taken in different, clear lighting conditions makes matching more resilient.

Use the **Forget** control on a person’s card to permanently remove their stored descriptors from the browser.

## Capture a face and teach a voice

With the camera open, choose **Capture face** to take a still from the viewfinder. Give the person a name and save it; that captured image is used to make the face descriptor, then discarded.

After saving a person, choose **Teach voice** on their card. The app listens for six seconds and turns the speech’s frequency pattern into a local voice cue. **Start listening** uses that cue to show the *likely* current speaker below the viewfinder. This is a lightweight, local similarity check—not a forensic-grade voice-identification system—so use it in a quiet setting and keep unknown or uncertain results as unknown.

## Notes

- Everything that handles your camera, selected photos, and face comparison runs in the browser. The app has no backend or user accounts.
- The voice feature keeps only a compact numerical sound profile; it does not save an audio recording.
- The externally hosted files are the object-detection and face-recognition model weights. They are fetched once at startup; your images are not sent to the model host.
- Camera permission requires using `http://localhost` (provided by `server.mjs`) rather than opening `index.html` directly from the file system.
