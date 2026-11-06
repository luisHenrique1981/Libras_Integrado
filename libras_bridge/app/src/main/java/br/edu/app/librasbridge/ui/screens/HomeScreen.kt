package br.edu.app.librasbridge.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
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
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp

@Composable
fun HomeScreen(
    onCameraMode: () -> Unit,
    onVoiceMode: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = "Libras Bridge V3",
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.Bold
        )
        Spacer(modifier = Modifier.height(12.dp))
        Card(modifier = Modifier.fillMaxWidth()) {
            Text(
                text = "V3 com dicionário JSON carregado em runtime, avatar em sequência de sinais e TTS para vocalizar o texto reconhecido pela câmera.",
                modifier = Modifier.padding(16.dp)
            )
        }
        Spacer(modifier = Modifier.height(24.dp))
        Button(onClick = onCameraMode, modifier = Modifier.fillMaxWidth()) {
            Text("Surdo → Ouvinte (câmera para texto + voz)")
        }
        Spacer(modifier = Modifier.height(16.dp))
        Button(onClick = onVoiceMode, modifier = Modifier.fillMaxWidth()) {
            Text("Ouvinte → Surdo (fala/texto para avatar)")
        }
    }
}
