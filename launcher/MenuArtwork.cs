using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.IO.Compression;
using System.Runtime.InteropServices;
using System.Text;

// Retail pixels are read only from the player's disc/install, never embedded
// in the launcher or written into the source tree.
sealed class MenuArtwork : IDisposable
{
    public Bitmap Logo, Sky, Screen, Skyline, Grass;
    public void Dispose()
    {
        foreach (Bitmap image in new[] { Logo, Sky, Screen, Skyline, Grass })
            if (image != null) image.Dispose();
    }

    public static MenuArtwork Load(string image, string installation)
    {
        Dictionary<string, byte[]> files;
        string shell = installation == null ? null : Path.Combine(installation, "files", "SHELL");
        if (shell != null && File.Exists(Path.Combine(shell, "SHELL.TEX")) && File.Exists(Path.Combine(shell, "SHELL.RTX"))) {
            files = new Dictionary<string, byte[]>();
            foreach (string name in new[] { "SHELL.TEX", "SHELL.RTX" }) {
                string path = Path.Combine(shell, name);
                if (new FileInfo(path).Length > 16*1024*1024) throw new InvalidDataException("Oversized menu resource.");
                files.Add(name, File.ReadAllBytes(path));
            }
        } else files = GameImage.ReadMenuFiles(image);
        byte[] textures = Inflate(files["SHELL.TEX"], "shell.tex");
        byte[] palettes = Inflate(files["SHELL.RTX"], "shell.rtx");
        var wanted = new HashSet<int>(new[] { 140, 141, 164, 174, 190, 192 });
        var colors = ReadPalettes(palettes, wanted);
        var decoded = new Dictionary<int, Bitmap>();
        MenuArtwork art = new MenuArtwork();
        try {
            // Six size classes followed by 128-byte DMA texture headers.
            if (textures.Length < 128 || U32(textures, 0) != 6 || U32(textures, 4) != 8)
                throw new InvalidDataException("Unsupported menu texture bank.");
            for (int pos = 128; pos <= textures.Length-128 && decoded.Count < wanted.Count;) {
                uint words = U32(textures, pos), id = U32(textures, pos+8), format = U32(textures, pos+12);
                if (words < 32 || words > (textures.Length-pos)/4)
                    throw new InvalidDataException("Invalid menu texture range.");
                int bytes = (int)words*4;
                if (wanted.Contains((int)id)) {
                    if (decoded.ContainsKey((int)id)) throw new InvalidDataException("Duplicate menu texture.");
                    decoded.Add((int)id, Decode(textures, pos+128, bytes-128, format, colors[(int)id]));
                }
                pos += bytes;
            }
            if (decoded.Count != wanted.Count) throw new InvalidDataException("The menu artwork is unavailable in this disc revision.");
            art.Logo = new Bitmap(512, 256, PixelFormat.Format32bppArgb);
            using (Graphics g = Graphics.FromImage(art.Logo)) {
                g.DrawImageUnscaled(decoded[140], 0, 0); g.DrawImageUnscaled(decoded[141], 256, 0);
            }
            // Clone ownership: temporary texture bitmaps are always disposed.
            art.Screen = new Bitmap(decoded[164]); art.Sky = new Bitmap(decoded[174]);
            art.Skyline = new Bitmap(decoded[190]); art.Grass = new Bitmap(decoded[192]);
            return art;
        } catch { art.Dispose(); throw; }
        finally { foreach (Bitmap bitmap in decoded.Values) bitmap.Dispose(); }
    }

    static uint U32(byte[] data, int offset)
    {
        if (offset < 0 || offset > data.Length-4) throw new InvalidDataException("Truncated menu resource.");
        return BitConverter.ToUInt32(data, offset);
    }
    static byte[] Inflate(byte[] data, string name)
    {
        // The game uses a ZIP local header with IE instead of PK. No filenames
        // become host paths, and expanded resources have a strict size limit.
        if (data.Length < 30 || !((data[0] == 'I' && data[1] == 'E') || (data[0] == 'P' && data[1] == 'K')) ||
            data[2] != 3 || data[3] != 4 || BitConverter.ToUInt16(data, 8) != 8 || (BitConverter.ToUInt16(data, 6)&9) != 0)
            throw new InvalidDataException("Unsupported compressed menu resource.");
        uint compressed = U32(data, 18), expanded = U32(data, 22);
        int nameLength = BitConverter.ToUInt16(data, 26), extraLength = BitConverter.ToUInt16(data, 28);
        int start = 30+nameLength+extraLength;
        if (start > data.Length || compressed > data.Length-start || expanded == 0 || expanded > 8*1024*1024 ||
            !String.Equals(Encoding.ASCII.GetString(data, 30, nameLength), name, StringComparison.OrdinalIgnoreCase))
            throw new InvalidDataException("Invalid compressed menu resource range.");
        byte[] result = new byte[(int)expanded];
        using (var input = new MemoryStream(data, start, (int)compressed, false))
        using (var unzip = new DeflateStream(input, CompressionMode.Decompress)) {
            int count = 0;
            while (count < result.Length) {
                int got = unzip.Read(result, count, result.Length-count);
                if (got == 0) throw new InvalidDataException("Truncated compressed menu artwork.");
                count += got;
            }
            if (unzip.ReadByte() != -1) throw new InvalidDataException("Oversized compressed menu artwork.");
        }
        if (Crc32(result) != U32(data, 14)) throw new InvalidDataException("Menu artwork checksum mismatch.");
        return result;
    }
    static uint Crc32(byte[] bytes)
    {
        uint[] table = new uint[256];
        for (uint i = 0; i < table.Length; ++i) {
            uint value = i;
            for (int j = 0; j < 8; ++j) value = (value >> 1) ^ ((value & 1) == 0 ? 0 : 0xedb88320u);
            table[i] = value;
        }
        uint crc = 0xffffffffu;
        foreach (byte value in bytes) crc = table[(crc ^ value) & 255] ^ (crc >> 8);
        return ~crc;
    }
    static Dictionary<int, int[]> ReadPalettes(byte[] data, HashSet<int> wanted)
    {
        var result = new Dictionary<int, int[]>();
        for (int pos = 0, index = 0; pos <= data.Length-16 && result.Count < wanted.Count; ++index) {
            uint words = U32(data, pos);
            if ((words != 20 && words != 260 && words != 28 && words != 268) || words > (data.Length-pos)/4)
                throw new InvalidDataException("Unsupported menu palette bank.");
            if (wanted.Contains(index+1)) {
                int size = words == 20 || words == 28 ? 16 : 256;
                int[] colors = new int[size];
                for (int i = 0; i < size; ++i) {
                    // GS CSM1 palette storage exchanges index bits 3 and 4.
                    int entry = size == 16 ? i : (i & ~24) | ((i & 8) << 1) | ((i & 16) >> 1);
                    int p = pos+16+entry*4;
                    int alpha = Math.Min(255, data[p+3]*2);
                    colors[i] = (alpha << 24) | (data[p] << 16) | (data[p+1] << 8) | data[p+2];
                }
                result.Add(index+1, colors);
            }
            pos += (int)words*4;
        }
        if (result.Count != wanted.Count) throw new InvalidDataException("Missing menu palettes.");
        return result;
    }
    static Bitmap Decode(byte[] bytes, int start, int length, uint format, int[] colors)
    {
        int width = 1 << (int)((format >> 8) & 15), height = 1 << (int)((format >> 12) & 15);
        int psm = (int)(format & 255);
        if (width > 256 || height > 256 || (psm != 0x13 && psm != 0x14) ||
            colors.Length != (psm == 0x13 ? 256 : 16) || length != width*height/(psm == 0x13 ? 1 : 2))
            throw new InvalidDataException("Unsupported menu texture format.");
        int[] pixels = new int[width*height];
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x) {
                int source = y*width+x;
                int index = psm == 0x13 ? bytes[start+source] : (bytes[start+source/2] >> ((source&1)*4)) & 15;
                // The main menu's UVs invert V relative to the stored texels.
                pixels[(height-1-y)*width+x] = colors[index];
            }
        var bitmap = new Bitmap(width, height, PixelFormat.Format32bppArgb);
        try {
            BitmapData locked = bitmap.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.WriteOnly, PixelFormat.Format32bppArgb);
            try { Marshal.Copy(pixels, 0, locked.Scan0, pixels.Length); }
            finally { bitmap.UnlockBits(locked); }
            return bitmap;
        } catch { bitmap.Dispose(); throw; }
    }
}
