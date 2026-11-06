package br.edu.app.librasbridge.ui.screens

import android.app.Application
import android.content.Context
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
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
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewmodel.compose.viewModel
import br.edu.app.librasbridge.avatar.LibrasAvatarPlayer
import br.edu.app.librasbridge.speech.SpeechToTextManager
import br.edu.app.librasbridge.viewmodel.EstadoVideo
import br.edu.app.librasbridge.viewmodel.FraseViewModel
import br.edu.app.librasbridge.viewmodel.MainViewModel

@Composable
fun VoiceToLibrasScreen(
    @Suppress("UNUSED_PARAMETER") viewModel: MainViewModel
) {
    VoiceToLibrasScreen()
}

@Composable
fun VoiceToLibrasScreen() {
    val app = LocalContext.current.applicationContext as Application
    val fraseViewModel: FraseViewModel = viewModel(
        factory = ViewModelProvider.AndroidViewModelFactory.getInstance(app)
    )
    VoiceToLibrasScreenContent(viewModel = fraseViewModel)
}

@Composable
fun VoiceToLibrasScreen(viewModel: FraseViewModel) {
    VoiceToLibrasScreenContent(viewModel = viewModel)
}

@Composable
private fun VoiceToLibrasScreenContent(viewModel: FraseViewModel) {
    val context = LocalContext.current
    val estado by viewModel.estado.collectAsState()

    var audioGranted by remember { mutableStateOf(hasAudioPermission(context)) }
    var manualText by remember { mutableStateOf("") }
    var isListening by remember { mutableStateOf(false) }
    var statusMessage by remember { mutableStateOf("Pronto para reconhecer fala ou texto.") }

    val launcher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { granted ->
        audioGranted = granted
        if (!granted) {
            statusMessage = "Permissão de áudio não concedida."
        }
    }

    val speech = remember {
        SpeechToTextManager(
            context = context,
            onTextResult = { text ->
                manualText = text
                statusMessage = "Texto reconhecido. Gerando avatar..."
                viewModel.montarFrase(text)
            },
            onStatus = { status ->
                statusMessage = status
            },
            onListeningChanged = { listening ->
                isListening = listening
            }
        )
    }

    DisposableEffect(Unit) {
        onDispose {
            speech.destroy()
        }
    }

    LaunchedEffect(Unit) {
        if (!audioGranted) {
            launcher.launch(android.Manifest.permission.RECORD_AUDIO)
        }
    }

    LaunchedEffect(estado) {
        statusMessage = when (val e = estado) {
            EstadoVideo.Inativo -> "Pronto para reconhecer fala ou texto."
            is EstadoVideo.Carregando -> "Processando sinais ${e.atual}/${e.total}..."
            is EstadoVideo.Pronto -> {
                "Sequência pronta: ${e.resultado.arquivosParaReproducao.size} vídeo(s) carregado(s)."
            }
            is EstadoVideo.ParcialmentePronto -> {
                "Sequência parcial: ${e.resultado.arquivosParaReproducao.size} vídeo(s) carregado(s); verifique os avisos."
            }
            is EstadoVideo.Erro -> "Falha ao gerar avatar."
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp)
    ) {
        Text(
            text = "Modo fala/texto → avatar Libras",
            style = MaterialTheme.typography.headlineSmall
        )

        Spacer(modifier = Modifier.height(8.dp))

        Card(modifier = Modifier.fillMaxWidth()) {
            Text(
                text = statusMessage,
                modifier = Modifier.padding(12.dp)
            )
        }

        Spacer(modifier = Modifier.height(12.dp))

        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            Button(
                onClick = {
                    if (audioGranted) {
                        speech.startListening()
                    } else {
                        statusMessage = "Permissão de áudio não concedida."
                    }
                }
            ) {
                Text(if (isListening) "Escutando..." else "Ouvir")
            }

            Button(
                onClick = { speech.stopListening() }
            ) {
                Text("Parar")
            }
        }

        Spacer(modifier = Modifier.height(12.dp))

        OutlinedTextField(
            value = manualText,
            onValueChange = { manualText = it },
            modifier = Modifier.fillMaxWidth(),
            label = { Text("Digite ou dite uma frase em português") }
        )

        Spacer(modifier = Modifier.height(8.dp))

        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            Button(
                onClick = {
                    val texto = manualText.trim()
                    if (texto.isBlank()) {
                        statusMessage = "Digite uma frase para traduzir."
                    } else {
                        statusMessage = "Gerando avatar..."
                        viewModel.montarFrase(texto)
                    }
                },
                enabled = manualText.isNotBlank()
            ) {
                Text("Gerar avatar")
            }

            Button(
                onClick = {
                    speech.stopListening()
                    manualText = ""
                    statusMessage = "Pronto para reconhecer fala ou texto."
                    viewModel.resetar()
                }
            ) {
                Text("Limpar")
            }
        }

        Spacer(modifier = Modifier.height(12.dp))

        Card(modifier = Modifier.fillMaxWidth()) {
            Column(modifier = Modifier.padding(12.dp)) {
                Text("Texto atual:")
                Text(manualText.ifBlank { "Aguardando fala ou texto..." })
            }
        }

        Spacer(modifier = Modifier.height(12.dp))

        LibrasAvatarPlayer(
            estado = estado,
            modifier = Modifier.fillMaxWidth()
        )

        Spacer(modifier = Modifier.height(12.dp))

        when (val e = estado) {
            EstadoVideo.Inativo -> {
                Card(modifier = Modifier.fillMaxWidth()) {
                    Text(
                        text = "Nenhuma animação gerada ainda.",
                        modifier = Modifier.padding(12.dp)
                    )
                }
            }

            is EstadoVideo.Carregando -> {
                Card(modifier = Modifier.fillMaxWidth()) {
                    Text(
                        text = "Aguarde. O resolvedor está processando a frase e validando os arquivos de vídeo.",
                        modifier = Modifier.padding(12.dp)
                    )
                }
            }

            is EstadoVideo.Pronto -> {
                Card(modifier = Modifier.fillMaxWidth()) {
                    Text(
                        text = "Tradução concluída com sucesso total.",
                        modifier = Modifier.padding(12.dp)
                    )
                }
            }

            is EstadoVideo.ParcialmentePronto -> {
                Card(modifier = Modifier.fillMaxWidth()) {
                    Column(modifier = Modifier.padding(12.dp)) {
                        Text(
                            text = "Tradução parcial com avisos explícitos",
                            color = MaterialTheme.colorScheme.error
                        )
                        Text(
                            text = e.resultado.mensagemErroParaUI(),
                            modifier = Modifier.padding(top = 8.dp),
                            color = MaterialTheme.colorScheme.error
                        )
                    }
                }
            }

            is EstadoVideo.Erro -> {
                Card(modifier = Modifier.fillMaxWidth()) {
                    Column(modifier = Modifier.padding(12.dp)) {
                        Text(
                            text = "Falha total",
                            color = MaterialTheme.colorScheme.error
                        )
                        Text(
                            text = e.mensagem,
                            modifier = Modifier.padding(top = 8.dp),
                            color = MaterialTheme.colorScheme.error
                        )
                    }
                }
            }
        }
    }
}

private fun hasAudioPermission(context: Context): Boolean {
    return ContextCompat.checkSelfPermission(
        context,
        android.Manifest.permission.RECORD_AUDIO
    ) == PackageManager.PERMISSION_GRANTED
}