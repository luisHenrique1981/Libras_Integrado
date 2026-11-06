package br.edu.app.librasbridge.data

import android.content.Context
import androidx.room.ColumnInfo
import androidx.room.Dao
import androidx.room.Database
import androidx.room.Entity
import androidx.room.Index
import androidx.room.PrimaryKey
import androidx.room.Query
import androidx.room.Room
import androidx.room.RoomDatabase
import kotlinx.coroutines.flow.Flow

@Entity(
    tableName = "palavras",
    indices = [
        Index(value = ["palavra"], name = "idx_palavra"),
        Index(value = ["letra"], name = "idx_letra"),
        Index(value = ["categoria"], name = "idx_categoria"),
        Index(value = ["ident"], name = "idx_ident", unique = true),
    ]
)
data class Palavra(
    @PrimaryKey(autoGenerate = true)
    @ColumnInfo(name = "id")
    val id: Long? = null,

    @ColumnInfo(name = "ident")
    val ident: Int? = null,

    @ColumnInfo(name = "palavra")
    val palavra: String,

    @ColumnInfo(name = "letra")
    val letra: String,

    @ColumnInfo(name = "descricao")
    val descricao: String? = null,

    @ColumnInfo(name = "exemplo")
    val exemplo: String? = null,

    @ColumnInfo(name = "libras_glosa")
    val librasGlosa: String? = null,

    @ColumnInfo(name = "categoria")
    val categoria: String? = null,

    @ColumnInfo(name = "categoria_id")
    val categoriaId: Int? = null,

    @ColumnInfo(name = "classe")
    val classe: String? = null,

    @ColumnInfo(name = "classe_id")
    val classeId: Int? = null,

    @ColumnInfo(name = "origem")
    val origem: String? = null,

    @ColumnInfo(name = "origem_id")
    val origemId: Int? = null,

    @ColumnInfo(name = "mao_id")
    val maoId: Int? = null,

    @ColumnInfo(name = "video_nome")
    val videoNome: String? = null,

    @ColumnInfo(name = "video_url")
    val videoUrl: String? = null,

    @ColumnInfo(name = "video_path")
    val videoPath: String? = null,

    @ColumnInfo(name = "video_sha1")
    val videoSha1: String? = null,

    @ColumnInfo(name = "image_nome")
    val imageNome: String? = null,

    @ColumnInfo(name = "status", defaultValue = "1")
    val status: Int? = null,

    @ColumnInfo(name = "criado_em")
    val criadoEm: Long = 0L,

    @ColumnInfo(name = "video_status", defaultValue = "'UNKNOWN'")
    val videoStatus: String? = null,

    @ColumnInfo(name = "missing_reason")
    val missingReason: String? = null,

    @ColumnInfo(name = "app_safe", defaultValue = "1")
    val appSafe: Int? = null,

    @ColumnInfo(name = "fallback_strategy", defaultValue = "'VIDEO'")
    val fallbackStrategy: String? = null,

    @ColumnInfo(name = "fallback_ref")
    val fallbackRef: String? = null,

    @ColumnInfo(name = "updated_em")
    val updatedEm: Long? = null,
)

@Dao
interface LibrasDao {

    @Query("SELECT * FROM palavras ORDER BY palavra ASC")
    fun observarTodas(): Flow<List<Palavra>>

    @Query(
        """
        SELECT * FROM palavras
        WHERE palavra LIKE :q || '%' COLLATE NOCASE
        ORDER BY palavra ASC
        LIMIT :limite
        """
    )
    fun buscarPorPrefixo(q: String, limite: Int = 50): Flow<List<Palavra>>

    @Query(
        """
        SELECT * FROM palavras
        WHERE palavra LIKE '%' || :q || '%' COLLATE NOCASE
        ORDER BY palavra ASC
        LIMIT :limite
        """
    )
    fun buscarFullText(q: String, limite: Int = 100): Flow<List<Palavra>>

    @Query("SELECT * FROM palavras WHERE UPPER(TRIM(palavra)) = UPPER(TRIM(:palavra)) LIMIT 1")
    suspend fun buscarExata(palavra: String): Palavra?

    @Query(
        """
        SELECT * FROM palavras
        WHERE COALESCE(status, 1) = 1
          AND COALESCE(app_safe, 1) = 1
          AND palavra IS NOT NULL
          AND TRIM(palavra) <> ''
        ORDER BY palavra ASC
        """
    )
    suspend fun buscarTodasAtivasParaResolver(): List<Palavra>

    @Query(
        """
        SELECT * FROM palavras
        WHERE COALESCE(status, 1) = 1
          AND COALESCE(app_safe, 1) = 1
          AND palavra IS NOT NULL
          AND TRIM(palavra) <> ''
          AND instr(TRIM(palavra), ' ') > 0
        ORDER BY palavra ASC
        """
    )
    suspend fun buscarExpressoesCompostas(): List<Palavra>

    @Query("SELECT * FROM palavras WHERE letra = :letra ORDER BY palavra ASC")
    fun porLetra(letra: String): Flow<List<Palavra>>

    @Query("SELECT * FROM palavras WHERE categoria = :categoria ORDER BY palavra ASC")
    fun porCategoria(categoria: String): Flow<List<Palavra>>

    @Query("SELECT DISTINCT letra FROM palavras ORDER BY letra ASC")
    suspend fun letrasDisponiveis(): List<String>

    @Query("SELECT DISTINCT categoria FROM palavras WHERE categoria IS NOT NULL ORDER BY categoria ASC")
    suspend fun categoriasDisponiveis(): List<String>

    @Query("SELECT COUNT(*) FROM palavras")
    suspend fun total(): Long

    @Query("SELECT COUNT(*) FROM palavras WHERE video_path IS NOT NULL")
    suspend fun totalComVideoLocal(): Long

    @Query("UPDATE palavras SET video_path = :path, video_sha1 = :sha1 WHERE id = :id")
    suspend fun atualizarVideoPath(id: Long, path: String, sha1: String?)
}

@Database(
    entities = [Palavra::class],
    version = 1,
    exportSchema = false,
)
abstract class LibrasDatabase : RoomDatabase() {

    abstract fun librasDao(): LibrasDao

    companion object {
        @Volatile
        private var INSTANCE: LibrasDatabase? = null

        fun getInstance(context: Context, fromAsset: Boolean = true): LibrasDatabase {
            return INSTANCE ?: synchronized(this) {
                INSTANCE ?: buildDatabase(context, fromAsset).also { INSTANCE = it }
            }
        }

        private fun buildDatabase(context: Context, fromAsset: Boolean): LibrasDatabase {
            val builder = Room.databaseBuilder(
                context.applicationContext,
                LibrasDatabase::class.java,
                "libras.db",
            )
            if (fromAsset) {
                builder.createFromAsset("databases/libras.db")
            }
            return builder.build()
        }
    }
}