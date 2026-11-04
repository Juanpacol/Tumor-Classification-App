from adapters.outbound.storage.filesystem_storage import FilesystemStorage
from adapters.outbound.storage.supabase_storage import SupabaseStorage
from ports.storage_port import StoragePort
from settings import Settings


def build_storage(settings: Settings) -> StoragePort:
    """Supabase Storage when its URL and key are set, the local disk otherwise."""
    if settings.supabase_url and settings.supabase_service_key:
        return SupabaseStorage(
            settings.supabase_url,
            settings.supabase_service_key,
            settings.supabase_storage_bucket,
        )
    return FilesystemStorage(settings.storage_dir)
