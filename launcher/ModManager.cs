using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.IO.Compression;
using System.Text.RegularExpressions;
using System.Windows.Forms;

// Declarative mod manifests. The roster module provides the second page;
// character modules fill individual slots with IDs from the installed game.
sealed class GameMod
{
    public string Id, Name, Description, FilePath, Kind, Requires;
    public int Slot = -1, GuestCharacter, Costume;
    public override string ToString() { return Name; }
    public static GameMod Read(string path)
    {
        string[] lines;
        if (String.Equals(Path.GetExtension(path),".zip",StringComparison.OrdinalIgnoreCase)) {
            using (var zip=ZipFile.OpenRead(path)) {
                ZipArchiveEntry manifest=null;
                foreach (var entry in zip.Entries) if (entry.FullName=="manifest.wotmmod") {
                    if (manifest!=null) throw new InvalidDataException("Package has duplicate manifests.");
                    manifest=entry;
                }
                if (manifest==null || manifest.Length>16384) throw new InvalidDataException("Package needs a small manifest.wotmmod at its root.");
                using (var reader=new StreamReader(manifest.Open())) {
                    var source=reader.ReadToEnd();
                    if (source.Length>16384) throw new InvalidDataException("Mod manifest is too large.");
                    lines=source.Split(new[] {"\r\n","\n"},StringSplitOptions.None);
                }
            }
        } else {
            if (new FileInfo(path).Length > 16384) throw new InvalidDataException("Mod manifest is too large.");
            lines=File.ReadAllLines(path);
        }
        var data = new Dictionary<string,string>(StringComparer.Ordinal);
        foreach (string raw in lines) {
            string line = raw.Trim(); if (line.Length == 0 || line.StartsWith("#")) continue;
            int split = line.IndexOf('=');
            if (split < 1) throw new InvalidDataException("Invalid mod manifest line.");
            string key = line.Substring(0,split).Trim(), value = line.Substring(split+1).Trim();
            if (key != "format" && key != "id" && key != "name" && key != "description" &&
                key != "kind" && key != "slot" && key != "guest" && key != "costume" && key != "requires")
                throw new InvalidDataException("Unsupported mod field: " + key);
            if (data.ContainsKey(key)) throw new InvalidDataException("Duplicate mod field: " + key);
            data.Add(key,value);
        }
        foreach (string key in new[] { "format", "id", "name", "description", "kind" })
            if (!data.ContainsKey(key)) throw new InvalidDataException("Missing mod field: " + key);
        if (data["format"] != "2") throw new InvalidDataException("Unsupported mod format version. This launcher requires format=2 mods.");
        if (!Regex.IsMatch(data["id"], "\\A[a-z0-9][a-z0-9-]{0,63}\\z")) throw new InvalidDataException("Invalid mod ID.");
        if (data["name"].Length == 0 || data["name"].Length > 80 || data["description"].Length > 1000)
            throw new InvalidDataException("Invalid mod name or description.");
        var mod = new GameMod { Id=data["id"], Name=data["name"], Description=data["description"],
            Kind=data["kind"], Requires=data.ContainsKey("requires") ? data["requires"] : "", FilePath=path };
        if (mod.Kind == "skip-intro") {
            if (data.ContainsKey("slot") || data.ContainsKey("guest") || data.ContainsKey("costume") || data.ContainsKey("requires"))
                throw new InvalidDataException("Skip Intro cannot declare character or dependency fields.");
        } else if (mod.Kind == "roster") {
            if (data.ContainsKey("slot") || data.ContainsKey("guest") || data.ContainsKey("costume") || data.ContainsKey("requires"))
                throw new InvalidDataException("A roster mod cannot declare a character slot.");
        } else if (mod.Kind == "character") {
            if (!data.ContainsKey("slot") || !data.ContainsKey("guest") || mod.Requires.Length == 0)
                throw new InvalidDataException("Character mods need slot, guest and requires fields.");
            if (!Int32.TryParse(data["slot"],out mod.Slot) || mod.Slot < 0 || mod.Slot >= 10)
                throw new InvalidDataException("Character slot must be 0 through 9.");
            if (!Int32.TryParse(data["guest"],out mod.GuestCharacter) ||
                Array.IndexOf(new[] {1,2,3,4,5,7,8,9,10,11,13},mod.GuestCharacter)<0)
                throw new InvalidDataException("This game revision has no supported assets for that character ID.");
            if (data.ContainsKey("costume") &&
                (!Int32.TryParse(data["costume"],out mod.Costume) || mod.Costume < 0 || mod.Costume > 3))
                throw new InvalidDataException("Character costume must be 0 through 3.");
            if (mod.GuestCharacter == 13 && mod.Costume > 1)
                throw new InvalidDataException("The boss has only two installed costumes.");
            if (!Regex.IsMatch(mod.Requires,"\\A[a-z0-9][a-z0-9-]{0,63}\\z"))
                throw new InvalidDataException("Invalid roster dependency ID.");
        } else throw new InvalidDataException("Unsupported mod kind: " + mod.Kind);
        return mod;
    }
}

sealed class ModSelection
{
    public bool HasRoster, SkipIntro;
    public readonly GameMod[] Slots = new GameMod[10];
}

sealed class ModManager : Form
{
    readonly string folder;
    readonly CheckedListBox list = new CheckedListBox();
    readonly TextBox detail = new TextBox();
    public HashSet<string> EnabledIds;
    public ModManager(string root, HashSet<string> enabled)
    {
        folder=Path.Combine(root,"mods"); Directory.CreateDirectory(folder);
        EnabledIds=new HashSet<string>(enabled,StringComparer.Ordinal);
        Text="War of the Monsters — Mods"; ClientSize=new Size(650,430);
        StartPosition=FormStartPosition.CenterParent; FormBorderStyle=FormBorderStyle.FixedDialog; MaximizeBox=false; MinimizeBox=false;
        BackColor=Color.FromArgb(17,22,30); ForeColor=Color.Gainsboro; Font=new Font("Segoe UI",10);
        var title=new Label {Text="Choose your enabled mods",Left=20,Top=16,Width=600,Height=28};Controls.Add(title);
        list.SetBounds(20,52,610,180);list.CheckOnClick=true;list.BackColor=Color.FromArgb(35,44,57);list.ForeColor=ForeColor;Controls.Add(list);
        detail.SetBounds(20,246,610,112);detail.Multiline=true;detail.ReadOnly=true;detail.ScrollBars=ScrollBars.Vertical;detail.BackColor=BackColor;detail.ForeColor=ForeColor;Controls.Add(detail);
        list.SelectedIndexChanged+=delegate { var m=list.SelectedItem as GameMod;detail.Text=m==null?"":m.Description; };
        var import=Button("Import mod...",20,375,140);import.Click+=delegate { Import(); };
        var open=Button("Open mods folder",174,375,165);open.Click+=delegate { Process.Start(new ProcessStartInfo(folder){UseShellExecute=true}); };
        var apply=Button("Apply",490,375,140);apply.Click+=delegate {
            var chosen=new HashSet<string>(StringComparer.Ordinal);foreach(GameMod m in list.CheckedItems)chosen.Add(m.Id);
            try { Resolve(folder,chosen); EnabledIds=chosen;DialogResult=DialogResult.OK;Close(); }
            catch(Exception ex){MessageBox.Show(this,ex.Message,"Mod selection",MessageBoxButtons.OK,MessageBoxIcon.Warning);}
        };
        Reload();
    }
    Button Button(string text,int x,int y,int width) {var b=new Button{Text=text,Left=x,Top=y,Width=width,Height=36,FlatStyle=FlatStyle.Flat};Controls.Add(b);return b;}
    void Reload()
    {
        list.Items.Clear();var errors=new List<string>();var ids=new HashSet<string>();
        foreach(string path in PackagePaths(folder)) try {
            var m=GameMod.Read(path);if(!ids.Add(m.Id))throw new InvalidDataException("Duplicate mod ID: "+m.Id);
            list.Items.Add(m,EnabledIds.Contains(m.Id));
        } catch(Exception ex) { errors.Add(Path.GetFileName(path)+": "+ex.Message); }
        detail.Text=errors.Count>0?String.Join(Environment.NewLine,errors):"Enable Expanded Roster, then choose the characters that fill its ten slots. Empty slots show question marks.";
    }
    void Import()
    {
        using(var picker=new OpenFileDialog {Title="Import War of the Monsters mod",Filter="War of the Monsters mods (*.zip)|*.zip",CheckFileExists=true}) {
            if(picker.ShowDialog(this)!=DialogResult.OK)return;
            try {
                var m=GameMod.Read(picker.FileName);string target=Path.Combine(folder,m.Id+".zip");
                if(File.Exists(target))throw new IOException("This mod is already installed. Remove or rename its existing manifest in the mods folder before importing a replacement.");
                foreach(GameMod item in list.Items)if(item.Id==m.Id)throw new IOException("This mod ID is already installed.");
                EnabledIds.Clear();foreach(GameMod item in list.CheckedItems)EnabledIds.Add(item.Id);
                File.Copy(picker.FileName,target,false);Reload();
            }catch(Exception ex){MessageBox.Show(this,ex.Message,"Unable to import mod",MessageBoxButtons.OK,MessageBoxIcon.Error);}
        }
    }
    public static ModSelection Resolve(string folder,HashSet<string> enabled)
    {
        var result=new ModSelection();var remaining=new HashSet<string>(enabled,StringComparer.Ordinal);
        if(enabled.Count==0)return result;
        var active=new Dictionary<string,GameMod>(StringComparer.Ordinal);
        if(Directory.Exists(folder))foreach(string path in PackagePaths(folder)) {
            GameMod m;
            try {m=GameMod.Read(path);}catch(InvalidDataException){continue;}
            if(!enabled.Contains(m.Id))continue;
            if(!remaining.Remove(m.Id))throw new InvalidDataException("Duplicate enabled mod ID: "+m.Id);
            active.Add(m.Id,m);
        }
        if(remaining.Count!=0)throw new InvalidDataException("An enabled mod is missing or invalid. Open Mods and choose your enabled mods again.");
        foreach(GameMod m in active.Values) if(m.Kind=="skip-intro") result.SkipIntro=true;
        int rosters=0;
        foreach(GameMod m in active.Values)if(m.Kind=="roster")++rosters;
        if(rosters>1)throw new InvalidDataException("Only one roster layout can be enabled.");
        result.HasRoster=rosters==1;
        foreach(GameMod m in active.Values)if(m.Kind=="character") {
            GameMod dependency;
            if(!active.TryGetValue(m.Requires,out dependency) || dependency.Kind!="roster")
                throw new InvalidDataException(m.Name+" requires the "+m.Requires+" roster mod.");
            if(result.Slots[m.Slot]!=null)throw new InvalidDataException("Two character mods use slot "+(m.Slot+1)+".");
            result.Slots[m.Slot]=m;
        }
        return result;
    }
    static IEnumerable<string> PackagePaths(string folder)
    {
        foreach(string path in Directory.GetFiles(folder,"*.zip")) yield return path;
        foreach(string path in Directory.GetFiles(folder,"*.wotmmod")) yield return path;
    }
}
