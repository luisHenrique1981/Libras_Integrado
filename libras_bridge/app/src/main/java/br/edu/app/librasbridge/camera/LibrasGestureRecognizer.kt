package br.edu.app.librasbridge.camera

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import androidx.camera.core.ImageProxy
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.gesturerecognizer.GestureRecognizer
import java.nio.ByteBuffer

class LibrasGestureRecognizer(private val context: Context) {

    private var recognizer: GestureRecognizer? = null
    val isReady: Boolean
        get() = recognizer != null

    fun setup(modelAssetName: String = "models/libras_gesture.task"): Boolean {
        return try {
            val baseOptions = BaseOptions.builder()
                .setModelAssetPath(modelAssetName)
                .build()

            val options = GestureRecognizer.GestureRecognizerOptions.builder()
                .setBaseOptions(baseOptions)
                .setRunningMode(RunningMode.IMAGE)
                .setNumHands(2)
                .build()

            recognizer = GestureRecognizer.createFromOptions(context, options)
            true
        } catch (_: Exception) {
            recognizer = null
            false
        }
    }

    fun recognize(imageProxy: ImageProxy): String {
        val localRecognizer = recognizer ?: return ""
        val bitmap = imageProxy.toBitmap()
        val mpImage = BitmapImageBuilder(bitmap).build()
        val result = localRecognizer.recognize(mpImage)
        return result.gestures().firstOrNull()?.firstOrNull()?.categoryName().orEmpty()
    }

    fun close() {
        recognizer?.close()
        recognizer = null
    }
}

private fun ImageProxy.toBitmap(): Bitmap {
    val yBuffer = planes[0].buffer
    val uBuffer = planes[1].buffer
    val vBuffer = planes[2].buffer

    val ySize = yBuffer.remaining()
    val uSize = uBuffer.remaining()
    val vSize = vBuffer.remaining()

    val nv21 = ByteArray(ySize + uSize + vSize)
    yBuffer.get(nv21, 0, ySize)
    vBuffer.get(nv21, ySize, vSize)
    uBuffer.get(nv21, ySize + vSize, uSize)

    val yuv = android.graphics.YuvImage(nv21, android.graphics.ImageFormat.NV21, width, height, null)
    val out = java.io.ByteArrayOutputStream()
    yuv.compressToJpeg(android.graphics.Rect(0, 0, width, height), 80, out)
    val imageBytes = out.toByteArray()
    val bmp = android.graphics.BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)

    val matrix = Matrix().apply {
        postRotate(imageInfo.rotationDegrees.toFloat())
    }

    return Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, matrix, true)
}
