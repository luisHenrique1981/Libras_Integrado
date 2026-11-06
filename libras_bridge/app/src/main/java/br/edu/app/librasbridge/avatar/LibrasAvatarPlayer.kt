package br.edu.app.librasbridge.avatar

import android.net.Uri
import android.util.Log
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.media3.common.MediaItem
import androidx.media3.exoplayer.ExoPlayer
import androidx.media3.ui.PlayerView
import br.edu.app.librasbridge.data.TokenResolucao
import br.edu.app.librasbridge.viewmodel.EstadoVideo

private const val TAG = "LibrasAvatarPlayer"

@Composable
fun LibrasAvatarPlayer(
    estado: EstadoVideo,
    modifier: Modifier = Modifier,
) {
    val context = LocalContext.current
    val player = remember(context) {
        ExoPlayer.Builder(context).build().apply {
            playWhenReady = true
        }
    }

    var mensagemUi by remember { mutableStateOf<String?>(null) }
    var labels by remember { mutableStateOf<List<String>>(emptyList()) }

    LaunchedEffect(estado) {
        when (val s = estado) {
            is EstadoVideo.Pronto -> {
                val arquivos = s.resultado.arquivosParaReproducao
                val mediaItems = arquivos.map { arquivo ->
                    MediaItem.fromUri(Uri.fromFile(arquivo))
                }

                labels = s.resultado.resolucoes
                    .filterIsInstance<TokenResolucao.Resolvido>()
                    .map { it.token }

                mensagemUi = null
                player.stop()
                player.clearMediaItems()

                if (mediaItems.isNotEmpty()) {
                    player.setMediaItems(mediaItems)
                    player.prepare()
                    player.play()
                    Log.i(TAG, "[PLAYER:READY] playlist carregada com ${mediaItems.size} item(ns).")
                } else {
                    mensagemUi = "Nenhum vídeo válido para reprodução."
                    Log.e(TAG, "[PLAYER:EMPTY] Estado Pronto sem arquivos.")
                }
            }

            is EstadoVideo.ParcialmentePronto -> {
                val arquivos = s.resultado.arquivosParaReproducao
                val mediaItems = arquivos.map { arquivo ->
                    MediaItem.fromUri(Uri.fromFile(arquivo))
                }

                labels = s.resultado.resolucoes
                    .filterIsInstance<TokenResolucao.Resolvido>()
                    .map { it.token }

                mensagemUi = s.resultado.mensagemErroParaUI()
                player.stop()
                player.clearMediaItems()

                if (mediaItems.isNotEmpty()) {
                    player.setMediaItems(mediaItems)
                    player.prepare()
                    player.play()
                    Log.w(TAG, "[PLAYER:PARTIAL] playlist parcial com ${mediaItems.size} item(ns).")
                } else {
                    Log.e(TAG, "[PLAYER:PARTIAL_EMPTY] Estado parcial sem arquivos reproduzíveis.")
                }
            }

            is EstadoVideo.Erro -> {
                labels = emptyList()
                mensagemUi = s.mensagem
                player.stop()
                player.clearMediaItems()
                Log.e(TAG, "[PLAYER:ERROR] ${s.mensagem}")
            }

            is EstadoVideo.Carregando -> {
                labels = emptyList()
                mensagemUi = "Processando sinais ${s.atual}/${s.total}..."
            }

            EstadoVideo.Inativo -> {
                labels = emptyList()
                mensagemUi = null
                player.stop()
                player.clearMediaItems()
                Log.d(TAG, "[PLAYER:IDLE]")
            }
        }
    }

    DisposableEffect(Unit) {
        onDispose {
            player.release()
            Log.d(TAG, "[PLAYER:RELEASE]")
        }
    }

    Column(modifier = modifier) {
        AndroidView(
            modifier = Modifier
                .fillMaxWidth()
                .height(260.dp),
            factory = { ctx ->
                PlayerView(ctx).apply {
                    useController = true
                    this.player = player
                }
            },
            update = { view ->
                view.player = player
            }
        )

        when (val s = estado) {
            is EstadoVideo.Carregando -> {
                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(top = 12.dp)
                ) {
                    Row(
                        modifier = Modifier.padding(12.dp),
                        horizontalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        CircularProgressIndicator(
                            modifier = Modifier.size(20.dp),
                            strokeWidth = 2.dp
                        )
                        Text("Processando sinais ${s.atual}/${s.total}")
                    }
                }
            }

            is EstadoVideo.Erro -> {
                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(top = 12.dp)
                ) {
                    Column(modifier = Modifier.padding(12.dp)) {
                        Text(
                            text = "Falha na tradução/reprodução",
                            style = MaterialTheme.typography.titleMedium,
                            color = MaterialTheme.colorScheme.error,
                        )
                        Text(
                            text = s.mensagem,
                            modifier = Modifier.padding(top = 8.dp),
                            color = MaterialTheme.colorScheme.error,
                        )
                    }
                }
            }

            else -> {
                if (!mensagemUi.isNullOrBlank() || labels.isNotEmpty()) {
                    Card(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(top = 12.dp)
                    ) {
                        Column(modifier = Modifier.padding(12.dp)) {
                            if (!mensagemUi.isNullOrBlank()) {
                                Text(
                                    text = mensagemUi!!,
                                    color = MaterialTheme.colorScheme.error,
                                )
                            }

                            if (labels.isNotEmpty()) {
                                Text(
                                    text = if (mensagemUi.isNullOrBlank()) {
                                        "Sequência de sinais carregada:"
                                    } else {
                                        "Sinais reproduzíveis encontrados:"
                                    },
                                    modifier = Modifier.padding(
                                        top = if (mensagemUi.isNullOrBlank()) 0.dp else 12.dp
                                    ),
                                )

                                Column(modifier = Modifier.padding(top = 8.dp)) {
                                    labels.forEach { label ->
                                        Text("• $label")
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Row(
            modifier = Modifier.padding(top = 12.dp),
            horizontalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Button(
                onClick = { player.seekToPreviousMediaItem() },
                enabled = player.mediaItemCount > 1,
            ) {
                Text("Anterior")
            }

            Button(
                onClick = { player.seekToNextMediaItem() },
                enabled = player.mediaItemCount > 1,
            ) {
                Text("Próximo")
            }
        }
    }
}