package br.edu.app.librasbridge.ui.screens

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import br.edu.app.librasbridge.camera.CameraFrameAnalyzer
import br.edu.app.librasbridge.camera.LibrasGestureRecognizer
import br.edu.app.librasbridge.speech.TextToSpeechManager
import br.edu.app.librasbridge.util.PermissionUtils
import br.edu.app.librasbridge.viewmodel.MainViewModel
import java.util.concurrent.Executors

@Composable
fun CameraTranslateScreen(viewModel: MainViewModel) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val state by viewModel.uiState.collectAsState()
    val previewView = remember { PreviewView(context) }
    val executor = remember { Executors.newSingleThreadExecutor() }
    val gestureRecognizer = remember { LibrasGestureRecognizer(context) }
    val ttsManager = remember {
        TextToSpeechManager(
            context = context,
            onStatus = { viewModel.setStatus(it) },
            onSpeakingChanged = { viewModel.setSpeaking(it) }
        )
    }
    var granted by remember { mutableStateOf(hasAllPermissions(context)) }

    val launcher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestMultiplePermissions()
    ) { result ->
        granted = result.values.all { it }
        if (!granted) {
            viewModel.setStatus("Permissões de câmera/microfone não concedidas.")
        }
    }

    LaunchedEffect(Unit) {
        if (!granted) launcher.launch(PermissionUtils.requiredPermissions)
        val ready = gestureRecognizer.setup()
        viewModel.setModelReady(ready)
    }

    DisposableEffect(Unit) {
        onDispose {
            executor.shutdown()
            gestureRecognizer.close()
            ttsManager.shutdown()
        }
    }

    LaunchedEffect(granted) {
        if (granted) {
            startCamera(
                context = context,
                lifecycleOwner = lifecycleOwner,
                previewView = previewView,
                analyzer = CameraFrameAnalyzer(gestureRecognizer) { raw, token ->
                    viewModel.appendRecognizedToken(raw, token)
                },
                executor = executor
            )
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
    ) {
        Text("Modo câmera: LIBRAS → texto", style = MaterialTheme.typography.headlineSmall)
        Spacer(modifier = Modifier.height(8.dp))
        Card(modifier = Modifier.fillMaxWidth()) {
            Text(
                text = state.statusMessage,
                modifier = Modifier.padding(12.dp)
            )
        }
        Spacer(modifier = Modifier.height(12.dp))
        AndroidView(
            factory = { previewView },
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f)
        )
        Spacer(modifier = Modifier.height(12.dp))
        Card(modifier = Modifier.fillMaxWidth()) {
            Column(modifier = Modifier.padding(12.dp)) {
                Text("Gesto bruto: ${state.recognizedGestureRaw.ifBlank { "-" }}")
                Text("Token: ${state.recognizedGestureToken.ifBlank { "-" }}")
                Spacer(modifier = Modifier.height(8.dp))
                Text("Frase montada:")
                Text(state.assembledSentence.ifBlank { "Aguardando sinais..." })
            }
        }
        Spacer(modifier = Modifier.height(12.dp))
        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            Button(onClick = { viewModel.clearCameraSentence() }) {
                Text("Limpar")
            }
            Button(onClick = { ttsManager.speak(state.assembledSentence) }, enabled = state.assembledSentence.isNotBlank()) {
                Text(if (state.isSpeaking) "Falando..." else "Falar")
            }
            Button(onClick = {
                val clipboard = context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
                clipboard.setPrimaryClip(ClipData.newPlainText("frase_libras", state.assembledSentence))
                viewModel.setStatus("Texto copiado para a área de transferência.")
            }, enabled = state.assembledSentence.isNotBlank()) {
                Text("Copiar")
            }
        }
    }
}

private fun hasAllPermissions(context: Context): Boolean {
    return PermissionUtils.requiredPermissions.all {
        ContextCompat.checkSelfPermission(context, it) == PackageManager.PERMISSION_GRANTED
    }
}

private fun startCamera(
    context: Context,
    lifecycleOwner: androidx.lifecycle.LifecycleOwner,
    previewView: PreviewView,
    analyzer: ImageAnalysis.Analyzer,
    executor: java.util.concurrent.ExecutorService
) {
    val providerFuture = ProcessCameraProvider.getInstance(context)
    providerFuture.addListener({
        val provider = providerFuture.get()

        val preview = Preview.Builder().build().also {
            it.surfaceProvider = previewView.surfaceProvider
        }

        val analysis = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()
            .also { it.setAnalyzer(executor, analyzer) }

        provider.unbindAll()
        provider.bindToLifecycle(
            lifecycleOwner,
            CameraSelector.DEFAULT_BACK_CAMERA,
            preview,
            analysis
        )
    }, ContextCompat.getMainExecutor(context))
}
