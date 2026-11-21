"""Exception khusus agar pesan error jelas (tidak ada hasil palsu)."""


class DashboardError(Exception):
    pass


class InvalidImageError(DashboardError):
    """File bukan gambar, kosong, atau corrupt."""


class ModelLoadError(DashboardError):
    """Model H5 gagal dimuat atau bentuknya tidak sesuai."""


class DatasetError(DashboardError):
    """Dataset kosong, struktur tidak dikenali, ZIP rusak, dll."""


class DatasetNotConfigured(DatasetError):
    """DATASET_PATH belum diisi."""


class UnknownLabelError(DatasetError):
    """Label/folder tidak ada di daftar kelas."""