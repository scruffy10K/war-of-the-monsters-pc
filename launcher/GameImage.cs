using System;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using System.Collections.Generic;

// Import a supported ISO into an extracted installation. The small index retains
// original byte locations so streaming consumers do not need a duplicate ISO.
static class GameImage
{
    const string Expected = "67AC3E4BF656F688767AD654DF7FE310317EFC4909D949AD32FD762825CF659E";
    static byte[] Read(FileStream stream, long offset, int length)
    {
        if (offset < 0 || length < 0 || offset > stream.Length - length)
            throw new InvalidDataException("The ISO is truncated or has an invalid directory.");
        byte[] bytes = new byte[length]; stream.Position = offset;
        for (int count = 0; count < length;) {
            int n = stream.Read(bytes,count,length-count);
            if (n == 0) throw new EndOfStreamException("The ISO could not be read completely.");
            count += n;
        }
        return bytes;
    }
    static string Hash(byte[] bytes)
    { using (SHA256 hash = SHA256.Create()) return BitConverter.ToString(hash.ComputeHash(bytes)).Replace("-",""); }
    // Read only these fixed artwork files, in memory. This works before an
    // installation exists and never copies retail assets into the launcher.
    public static Dictionary<string, byte[]> ReadMenuFiles(string image)
    {
        using (FileStream input = File.OpenRead(image)) {
            byte[] volume = Read(input, 16*2048, 2048);
            if (volume[0] != 1 || Encoding.ASCII.GetString(volume, 1, 5) != "CD001" || volume[6] != 1 ||
                BitConverter.ToUInt16(volume, 128) != 2048 || volume[156] < 34)
                throw new InvalidDataException("Unsupported ISO format for menu artwork.");
            uint extent, size;
            FindMenuEntry(input, BitConverter.ToUInt32(volume, 158), BitConverter.ToUInt32(volume, 166),
                "SHELL", true, out extent, out size);
            var files = new Dictionary<string, byte[]>();
            foreach (string name in new[] { "SHELL.TEX", "SHELL.RTX" }) {
                uint fileExtent, fileSize;
                FindMenuEntry(input, extent, size, name, false, out fileExtent, out fileSize);
                if (fileSize > 16*1024*1024) throw new InvalidDataException("Oversized menu resource in ISO.");
                files.Add(name, Read(input, (long)fileExtent*2048, (int)fileSize));
            }
            return files;
        }
    }
    static void FindMenuEntry(FileStream input, uint extent, uint size, string wanted, bool isDirectory, out uint foundExtent, out uint foundSize)
    {
        if (size == 0 || size > 16*1024*1024) throw new InvalidDataException("Invalid menu directory size.");
        byte[] directory = Read(input, (long)extent*2048, (int)size);
        for (int pos = 0; pos < directory.Length;) {
            int length = directory[pos];
            if (length == 0) { pos = (pos/2048+1)*2048; continue; }
            if (length < 34 || pos+length > directory.Length || pos%2048+length > 2048 || directory[pos+32] > length-33)
                throw new InvalidDataException("Invalid menu directory entry.");
            string name = Encoding.ASCII.GetString(directory, pos+33, directory[pos+32]).Split(';')[0];
            if (String.Equals(name, wanted, StringComparison.OrdinalIgnoreCase)) {
                byte flags = directory[pos+25];
                if ((flags & 128) != 0 || ((flags & 2) != 0) != isDirectory)
                    throw new InvalidDataException("Unsupported menu file entry.");
                foundExtent = BitConverter.ToUInt32(directory, pos+2);
                foundSize = BitConverter.ToUInt32(directory, pos+10);
                return;
            }
            pos += length;
        }
        throw new FileNotFoundException("Menu artwork not found in the selected ISO.");
    }
    public static string Install(string image, string root, Action<string> progress)
    {
        Prepare(image, Path.Combine(root,"SCUS_971.97"));
        string folder = Path.Combine(root,"game_data","install-" + DateTime.Now.ToString("yyyyMMdd-HHmmss") + "-" + Guid.NewGuid().ToString("N").Substring(0,8));
        Directory.CreateDirectory(folder);
        string assets=Path.Combine(folder,"files"); Directory.CreateDirectory(assets);
        var visited=new HashSet<uint>(); int files=0;
        var ranges=new List<DiscRange>();
        using (FileStream input=File.OpenRead(image)) {
            byte[] volume=Read(input,16*2048,2048);
            ExtractDirectory(input,BitConverter.ToUInt32(volume,158),BitConverter.ToUInt32(volume,166),assets,visited,ref files,0,progress,ranges,"files/");
            WriteDiscIndex(input,folder,ranges,progress);
        }
        // Verify the extracted executable; publish the activation marker last.
        PrepareInstalled(folder,Path.Combine(root,"SCUS_971.97"));
        File.WriteAllText(Path.Combine(folder,"installation.txt"),"WOTM-INSTALL-2\n" + Expected + "\nFiles=" + files +
            "\nIndex=" + HashFile(Path.Combine(folder,"disc.index")) + "\nMetadata=" + HashFile(Path.Combine(folder,"disc.meta")) + "\n");
        return folder;
    }
    static void ExtractDirectory(FileStream input,uint extent,uint size,string folder,HashSet<uint> visited,ref int files,int depth,Action<string> progress,List<DiscRange> ranges,string relative)
    {
        if (depth>16 || visited.Count>4096 || !visited.Add(extent) || size>16*1024*1024)
            throw new InvalidDataException("Unsupported or cyclic ISO directory.");
        byte[] directory=Read(input,(long)extent*2048,(int)size);
        for(int pos=0;pos<directory.Length;) {
            int length=directory[pos];
            if(length==0) { pos=(pos/2048+1)*2048; continue; }
            if(length<34 || pos+length>directory.Length || pos%2048+length>2048 || directory[pos+32]>length-33)
                throw new InvalidDataException("Invalid ISO directory entry.");
            string name=Encoding.ASCII.GetString(directory,pos+33,directory[pos+32]).Split(';')[0];
            uint lba=BitConverter.ToUInt32(directory,pos+2), bytes=BitConverter.ToUInt32(directory,pos+10);
            byte flags=directory[pos+25]; pos+=length;
            if(name=="\0" || name=="\u0001") continue;
            if(String.IsNullOrEmpty(name) || name=="." || name==".." || name.IndexOfAny(Path.GetInvalidFileNameChars())>=0 || name.EndsWith(".") || name.EndsWith(" "))
                throw new InvalidDataException("Unsafe filename in ISO.");
            if((flags&128)!=0) throw new InvalidDataException("Multi-extent ISO files are unsupported.");
            string path=Path.Combine(folder,name);
            if((flags&2)!=0) { Directory.CreateDirectory(path); ExtractDirectory(input,lba,bytes,path,visited,ref files,depth+1,progress,ranges,relative+name+"/"); }
            else {
                if(++files>100000 || (long)lba*2048>input.Length-bytes) throw new InvalidDataException("Invalid ISO file range.");
                if(bytes!=0) ranges.Add(new DiscRange { Offset=(long)lba*2048, Length=bytes, Path=relative+name });
                progress("Extracting " + name + " (" + files + " files)");
                input.Position=(long)lba*2048;
                using(FileStream output=new FileStream(path,FileMode.CreateNew,FileAccess.Write)) {
                    byte[] buffer=new byte[1024*1024]; long left=bytes;
                    while(left>0) { int got=input.Read(buffer,0,(int)Math.Min(buffer.Length,left)); if(got==0) throw new EndOfStreamException(); output.Write(buffer,0,got); left-=got; }
                }
            }
        }
    }
    sealed class DiscRange
    {
        public long Offset,Length,FileOffset;
        public string Path;
    }
    static string HashFile(string path)
    {
        using(var input=File.OpenRead(path)) using(var hash=SHA256.Create())
            return BitConverter.ToString(hash.ComputeHash(input)).Replace("-","");
    }
    static void WriteDiscIndex(FileStream input,string folder,List<DiscRange> files,Action<string> progress)
    {
        if(input.Length<32768 || input.Length>8L*1024*1024*1024 || input.Length%2048!=0)
            throw new InvalidDataException("Unsupported disc image size.");
        files.Sort((a,b)=>a.Offset.CompareTo(b.Offset));
        var all=new List<DiscRange>(files);
        long cursor=0;
        byte[] buffer=new byte[65536];
        progress("Verifying disc layout and preserving streaming metadata...");
        using(var metadata=new FileStream(Path.Combine(folder,"disc.meta"),FileMode.CreateNew,FileAccess.Write)) {
            // Preserve every nonzero byte outside file extents, including ISO
            // directories and sector tails. Only verified zero gaps are implicit.
            for(int i=0;i<=files.Count;++i) {
                long stop=i==files.Count?input.Length:files[i].Offset;
                if(stop<cursor) throw new InvalidDataException("Overlapping ISO file extents are unsupported.");
                input.Position=cursor;
                while(cursor<stop) {
                    int length=(int)Math.Min(buffer.Length,stop-cursor),got=0;
                    while(got<length) { int n=input.Read(buffer,got,length-got); if(n==0) throw new EndOfStreamException(); got+=n; }
                    bool nonzero=false;
                    for(int b=0;b<length;++b) if(buffer[b]!=0) { nonzero=true; break; }
                    if(nonzero) {
                        if(all.Count>=100000) throw new InvalidDataException("The disc layout is too complex.");
                        all.Add(new DiscRange { Offset=cursor,Length=length,FileOffset=metadata.Position,Path="disc.meta" });
                        metadata.Write(buffer,0,length);
                    }
                    cursor+=length;
                }
                if(i<files.Count) cursor=files[i].Offset+files[i].Length;
            }
        }
        all.Sort((a,b)=>a.Offset.CompareTo(b.Offset));
        using(var output=new BinaryWriter(new FileStream(Path.Combine(folder,"disc.index"),FileMode.CreateNew,FileAccess.Write))) {
            output.Write(Encoding.ASCII.GetBytes("WOTMDI02"));
            output.Write(input.Length); output.Write(all.Count); output.Write(0);
            foreach(var range in all) {
                byte[] path=Encoding.UTF8.GetBytes(range.Path);
                output.Write(range.Offset); output.Write(range.Length); output.Write(range.FileOffset);
                output.Write(path.Length); output.Write(path);
            }
            if(output.BaseStream.Length>8*1024*1024) throw new InvalidDataException("Oversized installed disc index.");
        }
    }
    public static string InstallationImage(string folder)
    {
        string marker=Path.Combine(folder,"installation.txt");
        if(!File.Exists(marker)) return null;
        if(new FileInfo(marker).Length>4096) throw new InvalidDataException("Invalid installation receipt.");
        string[] lines=File.ReadAllLines(marker);
        if(lines.Length<2 || lines[1]!=Expected) return null;
        if(lines[0]=="WOTM-INSTALL-1") {
            string legacy=Path.Combine(folder,"disc.iso");
            return File.Exists(legacy)?legacy:null;
        }
        if(lines.Length!=5 || lines[0]!="WOTM-INSTALL-2") return null;
        string index=Path.Combine(folder,"disc.index"),meta=Path.Combine(folder,"disc.meta");
        if(!File.Exists(index) || !File.Exists(meta) || new FileInfo(index).Length>8*1024*1024 ||
            new FileInfo(meta).Length>8L*1024*1024*1024 ||
            lines[3]!="Index="+HashFile(index) || lines[4]!="Metadata="+HashFile(meta))
            throw new InvalidDataException("Installed disc metadata is missing or damaged. Install Game again from your ISO.");
        return index;
    }
    public static void PrepareInstalled(string folder,string destination)
    {
        string executable=Path.Combine(folder,"files","SCUS_971.97");
        if(!File.Exists(executable) || new FileInfo(executable).Length!=7414808)
            throw new InvalidDataException("The installed game executable is missing or damaged. Install Game again.");
        WriteExecutable(File.ReadAllBytes(executable),destination);
    }
    public static void Prepare(string image, string destination)
    {
        byte[] executable = null;
        using (FileStream stream = File.OpenRead(image)) {
            byte[] volume = Read(stream,16*2048,2048);
            if (volume[0] != 1 || Encoding.ASCII.GetString(volume,1,5) != "CD001" || volume[6] != 1)
                throw new InvalidDataException("Choose a standard ISO image of the supported US retail game (SCUS-97197).");
            if (BitConverter.ToUInt16(volume,128) != 2048 || volume[156] < 34)
                throw new InvalidDataException("Unsupported ISO directory format.");
            uint extent = BitConverter.ToUInt32(volume,158), size = BitConverter.ToUInt32(volume,166);
            if (size == 0 || size > 16*1024*1024) throw new InvalidDataException("Invalid ISO root directory size.");
            byte[] directory = Read(stream,(long)extent*2048,(int)size);
            for (int pos=0; pos<directory.Length;) {
                int length=directory[pos];
                if (length==0) { pos=(pos/2048+1)*2048; continue; }
                if (length<34 || pos+length>directory.Length || pos%2048+length>2048 || directory[pos+32]>length-33)
                    throw new InvalidDataException("Invalid ISO file entry.");
                string name=Encoding.ASCII.GetString(directory,pos+33,directory[pos+32]).Split(';')[0];
                if (name == "SCUS_971.97") {
                    if ((directory[pos+25] & 0x82) != 0) throw new InvalidDataException("Unsupported executable entry in ISO.");
                    uint bytes=BitConverter.ToUInt32(directory,pos+10);
                    if (bytes != 7414808) throw new InvalidDataException("This ISO contains a different game revision.");
                    executable=Read(stream,(long)BitConverter.ToUInt32(directory,pos+2)*2048,(int)bytes);
                    break;
                }
                pos+=length;
            }
        }
        WriteExecutable(executable,destination);
    }
    static void WriteExecutable(byte[] executable,string destination)
    {
        if (executable==null || Hash(executable)!=Expected)
            throw new InvalidDataException("This port needs the US retail War of the Monsters executable (SCUS-97197). The selected ISO is a different revision or is modified.");
        if (File.Exists(destination)) {
            if (new FileInfo(destination).Length != executable.Length || Hash(File.ReadAllBytes(destination)) != Expected)
                throw new IOException("An incompatible executable already exists at " + destination + ". Move it aside before importing this disc.");
            return;
        }
        // CreateNew prevents silently replacing an existing user file.
        using (FileStream output = new FileStream(destination,FileMode.CreateNew,FileAccess.Write))
            output.Write(executable,0,executable.Length);
    }
}
