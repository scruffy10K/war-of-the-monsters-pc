using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

sealed class Launcher : Form
{
    static readonly string Root = ResolveRoot();
    static readonly string SettingsPath = Path.Combine(Root, "launcher", "settings.ini");

    static string ResolveRoot()
    {
        string directory = AppDomain.CurrentDomain.BaseDirectory.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        return String.Equals(Path.GetFileName(directory), "launcher", StringComparison.OrdinalIgnoreCase)
            ? Path.GetFullPath(Path.Combine(directory, "..")) : Path.GetFullPath(directory);
    }
    static string GameExecutable()
    {
        string packaged = Path.Combine(Root, "game", "War of the Monsters.exe");
        if (File.Exists(packaged)) return packaged;
        string installed = GameBuild.InstalledExecutable(Root);
        if (installed != null) return installed;
        string directory = Path.Combine(Root, "build64", "ps2xRuntime");
        string source = Path.Combine(directory, "ps2EntryRunner.exe");
        string legacy = Path.Combine(directory, "War of the Monsters.exe");
        if (File.Exists(source) && (!File.Exists(legacy) || File.GetLastWriteTimeUtc(source) != File.GetLastWriteTimeUtc(legacy)))
            File.Copy(source, legacy, true);
        if (!File.Exists(legacy)) throw new IOException("The game runtime is missing. Extract the complete Windows download, including its game folder, before playing.");
        return legacy;
    }
    readonly Dictionary<string, ComboBox> choices = new Dictionary<string, ComboBox>();
    readonly ToolTip help = new ToolTip { AutoPopDelay = 12000, InitialDelay = 300, ReshowDelay = 100, ShowAlways = true };
    readonly Label status = new Label();
    readonly Button play = new MovieButton { Primary = true };
    readonly TextBox iso = new TextBox();
    Process game;
    string installed = "";
    HashSet<string> enabledMods = new HashSet<string>(StringComparer.Ordinal);
    bool installing;
    StreamWriter log;
    string rendererFailure;
    readonly object logLock = new object();
    readonly TimerHolder poll = new TimerHolder();
    readonly Color muted = MovieTheme.Muted;
    MenuArtwork artwork;
    float artworkScale = 1;

    [DllImport("user32.dll")] static extern bool SetProcessDPIAware();
    [STAThread] static int Main(string[] args)
    {
        SetProcessDPIAware();
        Application.EnableVisualStyles();
        Application.SetCompatibleTextRenderingDefault(false);
        bool first;
        using (Mutex mutex = new Mutex(true, "Local\\WotmGameLauncher", out first))
        {
            if (!first) { MessageBox.Show("The launcher is already open.", "War of the Monsters"); return 0; }
            using (Launcher form = new Launcher())
            {
                if (args.Length == 2 && args[0] == "--preview")
                {
                    form.Show(); Application.DoEvents(); form.Refresh(); Application.DoEvents();
                    using (Bitmap bitmap = new Bitmap(form.Width, form.Height))
                    { form.DrawToBitmap(bitmap, new Rectangle(0,0,form.Width,form.Height)); bitmap.Save(args[1]); }
                    form.Close(); return 0;
                }
                Application.Run(form);
            }
        }
        return 0;
    }

    Launcher()
    {
        Text = "War of the Monsters";
        ClientSize = new Size(1120, 780);
        FormBorderStyle = FormBorderStyle.FixedSingle;
        MaximizeBox = false;
        StartPosition = FormStartPosition.CenterScreen;
        AutoScaleDimensions = new SizeF(96,96);
        AutoScaleMode = AutoScaleMode.Dpi;
        AutoScroll = true;
        BackColor = MovieTheme.Ink;
        ForeColor = MovieTheme.Paper;
        Font = new Font("Segoe UI", 10);
        DoubleBuffered = true;
        Section("GRAPHICS & PERFORMANCE",112,225);
        Section("DISPLAY",592,225);

        var modes = new List<string> { "1280x720", "1600x900", "1920x1080", "2560x1440", "3840x2160" };
        string desktop = Screen.PrimaryScreen.Bounds.Width + "x" + Screen.PrimaryScreen.Bounds.Height;
        if (!modes.Contains(desktop)) modes.Add(desktop);
        AddChoice("Graphics API", "renderer", new[] { "OpenGL", "Direct3D 12" }, "OpenGL",112,250);
        AddChoice("Internal resolution", "scale", new[] { "1x - 640 x 448", "2x - 1280 x 896", "3x - 1920 x 1344", "4x - 2560 x 1792", "6x - 3840 x 2688", "8x - 5120 x 3584" }, "3x - 1920 x 1344",112,300);
        AddChoice("Anti-aliasing", "aa", new[] { "Off", "FXAA" }, "Off",112,350);
        AddChoice("Display refresh cap", "fps", new[] { "60", "90", "120", "144", "165", "240", "360", "Unlimited" }, "60",112,400);
        AddChoice("VSync", "vsync", new[] { "Off", "On" }, "Off",112,450);
        AddChoice("High-refresh smoothing", "interpolation", new[] { "Off", "Native motion (D3D12, experimental)" }, "Off",112,500);
        AddChoice("Resolution", "window", modes.ToArray(), "1920x1080",592,250);
        AddChoice("Display mode", "mode", new[] { "Windowed", "Borderless", "Fullscreen" }, "Windowed",592,300);
        AddChoice("Aspect ratio", "aspect", new[] { "Original", "Widescreen (16:9)" }, "Original",592,350);
        Section("CONTROLS",592,409);
        AddChoice("Controller prompts", "glyphs", new[] { "Automatic", "PlayStation", "Xbox" }, "Automatic",592,436);
        AddChoice("Camera controls", "camera", new[] { "Original", "Modern orbit" }, "Original",592,486);
        Section("GAME DISC IMAGE",112,562);
        iso.SetBounds(112,590,590,30); iso.ReadOnly = true;
        iso.BorderStyle = BorderStyle.FixedSingle;
        iso.BackColor = MovieTheme.Field; iso.ForeColor = ForeColor;
        iso.Text = String.Empty; Controls.Add(iso);
        Button browse = ButtonAt("BROWSE...",714,588,132,34);
        browse.Click += delegate {
            using (OpenFileDialog picker = new OpenFileDialog()) {
                picker.Title = "Select your War of the Monsters ISO";
                picker.Filter = "Disc images (*.iso)|*.iso";
                picker.CheckFileExists = true;
                if (File.Exists(iso.Text)) picker.FileName = iso.Text;
                if (picker.ShowDialog(this) == DialogResult.OK) {
                    iso.Text = picker.FileName; installed = ""; RefreshArtwork();
                    status.Text = "ISO selected. Choose Install Game to set up the game on this PC.";
                }
            }
        };
        Button install = ButtonAt("INSTALL GAME",858,588,150,34);
        install.Click += delegate { InstallData(); };
        help.SetToolTip(iso,"Your supported US retail game ISO. After installation the original ISO is no longer required.");
        help.SetToolTip(browse,"Choose your War of the Monsters ISO file.");
        help.SetToolTip(install,"Copy your ISO and extract its game files into this folder. The game is already compiled: no tool downloads or developer software are needed. Keep about 3 GB free for installation.");
        status.SetBounds(112,627,896,35); status.BackColor=Color.Transparent; status.ForeColor=muted; Controls.Add(status);
        Button defaults = ButtonAt("DEFAULTS",112,678,139,43);
        defaults.Click += delegate { Defaults(); status.Text = "Defaults restored. Play saves your settings."; };
        Button save = ButtonAt("SAVE SETTINGS",263,678,169,43);
        save.Click += delegate { try { Save(); status.Text = "Settings saved."; } catch (Exception ex) { Error(ex); } };
        Button mods = ButtonAt("MODS...",444,678,159,43);
        mods.Click += delegate {
            using (var manager = new ModManager(Root, enabledMods)) {
                if (manager.ShowDialog(this) == DialogResult.OK) {
                    enabledMods = manager.EnabledIds;
                    status.Text = enabledMods.Count == 0 ? "No mods enabled." : "Roster mods selected for the next launch.";
                }
            }
        };
        help.SetToolTip(defaults,"Restore the default launcher settings and disable mods. Your installed game data is kept.");
        help.SetToolTip(save,"Save these settings without starting the game.");
        help.SetToolTip(mods,"Import and enable separately downloaded ZIP mods. No mods are included with the main game download.");
        Button about = ButtonAt("ABOUT",615,678,133,43);
        about.Click += delegate {
            MessageBox.Show(this,
                "War of the Monsters PC preview\nUnofficial bring-your-own-ISO port.\n\n" +
                "Based on PS2Recomp (GPL version 3).\nThis software uses libraries from the FFmpeg project under the LGPLv2.1.\n\n" +
                "Credits, license texts and source information are in THIRD_PARTY_NOTICES.md and the licenses folder.",
                "About War of the Monsters",MessageBoxButtons.OK,MessageBoxIcon.Information);
        };
        help.SetToolTip(about,"View the port's credits and open-source library notices.");
        play.Text = "PLAY"; play.SetBounds(760,675,248,49);
        StyleButton(play); play.Font = new Font(Font.FontFamily,15,FontStyle.Bold);
        play.Click += delegate { Launch(); }; Controls.Add(play);
        help.SetToolTip(play,"Start War of the Monsters with your selected settings and mods.");
        choices["mode"].SelectedIndexChanged += delegate { choices["window"].Enabled = choices["mode"].Text != "Borderless"; };
        choices["interpolation"].Enabled=choices["renderer"].SelectedIndex==1;
        choices["renderer"].SelectedIndexChanged += delegate { choices["interpolation"].Enabled=choices["renderer"].SelectedIndex==1; };
        LoadSettings();
        RefreshArtwork();
        choices["window"].Enabled = choices["mode"].Text != "Borderless";
        poll.Timer.Interval = 500;
        poll.Timer.Tick += delegate {
            if (game == null || !game.HasExited) return;
            game.WaitForExit();
            int code = game.ExitCode;
            lock (logLock) { if (log != null) { log.Dispose(); log = null; } }
            game.Dispose(); game = null; play.Enabled = true;
            status.Text = rendererFailure != null ? "Direct3D 12 stopped. Try OpenGL; details: wotm_last.log" : code == 0 ? "Ready to play." : "Game exited with code " + code + ". Details: wotm_last.log";
            Show(); WindowState = FormWindowState.Normal;
        };
        poll.Timer.Start();
        FormClosing += delegate(object sender, FormClosingEventArgs e) {
            if (installing) { e.Cancel = true; return; }
            if (game != null && !game.HasExited) { e.Cancel = true; Hide(); }
        };
        FormClosed += delegate { poll.Timer.Dispose(); help.Dispose(); if (artwork != null) artwork.Dispose(); };
        status.Text = InstalledImage() != null ? "Installed game data ready. Choose your settings and play." : "Select your ISO, then choose Install Game.";
    }

    protected override void OnShown(EventArgs e)
    {
        base.OnShown(e);
        MovieTheme.DarkTitle(this);
        artworkScale = ClientSize.Width/1120f;
        AutoScrollMinSize = ClientSize;
        Rectangle available = Screen.FromControl(this).WorkingArea;
        int chromeWidth = Width-ClientSize.Width, chromeHeight = Height-ClientSize.Height;
        Size fit = new Size(Math.Min(ClientSize.Width,available.Width-chromeWidth-24),
            Math.Min(ClientSize.Height,available.Height-chromeHeight-24));
        if (fit != ClientSize) { ClientSize = fit; CenterToScreen(); }
        Invalidate();
    }
    protected override void OnPaint(PaintEventArgs e)
    {
        base.OnPaint(e);
        var saved = e.Graphics.Save();
        e.Graphics.TranslateTransform(AutoScrollPosition.X,AutoScrollPosition.Y);
        e.Graphics.ScaleTransform(artworkScale,artworkScale);
        MovieTheme.PaintScene(e.Graphics,artwork);
        e.Graphics.Restore(saved);
    }
    void RefreshArtwork()
    {
        MenuArtwork next = null;
        string cached = InstalledImage();
        string image = cached ?? iso.Text;
        if (File.Exists(image)) {
            try { next = MenuArtwork.Load(image, cached == null ? null : Path.GetDirectoryName(cached)); }
            catch (InvalidDataException) { }
            catch (IOException) { }
            catch (UnauthorizedAccessException) { }
            catch (ArgumentException) { }
        }
        MenuArtwork old = artwork; artwork = next;
        if (old != null) old.Dispose();
        Invalidate();
    }
    Label LabelAt(string text, int x, int y, int width, int height)
    {
        Label l = new Label { Text=text,BackColor=Color.Transparent,UseMnemonic=false };
        l.SetBounds(x,y,width,height); Controls.Add(l); return l;
    }
    void Section(string text,int x,int y)
    {
        Label label=LabelAt(text,x,y,414,22);
        label.Font=new Font(Font.FontFamily,9,FontStyle.Bold); label.ForeColor=MovieTheme.Gold;
    }
    void StyleButton(Button b)
    { b.FlatStyle=FlatStyle.Flat; b.FlatAppearance.BorderSize=0; b.BackColor=MovieTheme.Field; b.ForeColor=ForeColor; }
    Button ButtonAt(string text, int x, int y, int w, int h)
    {
        Button b=new MovieButton { Text=text,Font=new Font(Font.FontFamily,9,FontStyle.Bold) };
        b.SetBounds(x,y,w,h); StyleButton(b); Controls.Add(b); return b;
    }
    void AddChoice(string label, string key, string[] values, string initial, int x, int y)
    {
        Label caption=LabelAt(label,x,y,377,22);
        Label question=new MovieHelp(); question.Font=new Font(Font.FontFamily,9,FontStyle.Bold);
        question.SetBounds(x+391,y,22,22); Controls.Add(question);
        string description=HelpText(key);
        help.SetToolTip(caption,description); help.SetToolTip(question,description);
        ComboBox box=new ComboBox { DropDownStyle=ComboBoxStyle.DropDownList,FlatStyle=FlatStyle.Flat,
            BackColor=MovieTheme.Field,ForeColor=ForeColor,Tag=initial,AccessibleName=label };
        box.DrawMode=DrawMode.OwnerDrawFixed; box.ItemHeight=22;
        box.DrawItem += delegate(object sender,DrawItemEventArgs e) {
            bool selected=(e.State&DrawItemState.Selected)!=0;
            using(var brush=new SolidBrush(selected?MovieTheme.Panel:box.BackColor)) e.Graphics.FillRectangle(brush,e.Bounds);
            if(e.Index>=0) TextRenderer.DrawText(e.Graphics,box.Items[e.Index].ToString(),box.Font,
                new Rectangle(e.Bounds.X+8,e.Bounds.Y,e.Bounds.Width-8,e.Bounds.Height),
                !box.Enabled?muted:selected?MovieTheme.Gold:box.ForeColor,
                TextFormatFlags.VerticalCenter | TextFormatFlags.Left | TextFormatFlags.EndEllipsis);
            if((e.State&DrawItemState.Focus)!=0) e.DrawFocusRectangle();
        };
        box.SetBounds(x,y+23,416,30); box.Items.AddRange(values); box.SelectedItem=initial;
        choices.Add(key,box); Controls.Add(box); help.SetToolTip(box,description);
        if(key=="renderer") box.SelectedIndexChanged += delegate {
            status.Text=box.SelectedIndex==1 ? "Direct3D 12 rendering. FXAA is available; native smoothing is optional." : "OpenGL rendering. FXAA is available.";
        };
    }
    static string HelpText(string key)
    {
        switch(key) {
            case "renderer": return "OpenGL and Direct3D 12 render the live game, textures, effects and movies. Both support FXAA and widescreen. Optional native motion smoothing is experimental and available only with Direct3D 12.";
            case "window": return "Window size or fullscreen resolution. Borderless uses your desktop resolution.";
            case "mode": return "Windowed has borders; Borderless fills the desktop; Fullscreen requests the selected display resolution.";
            case "aspect": return "Widescreen shows more of the world on the left and right during single-player gameplay. HUD proportions are preserved. Menus, movies and split-screen use the original aspect ratio. Works with OpenGL and Direct3D 12.";
            case "scale": return "Resolution used to draw the game internally. Higher values sharpen edges but use more GPU time and memory.";
            case "aa": return "FXAA smooths edges in the finished image on OpenGL or Direct3D 12. It can soften text and HUD details. Off keeps the original sharpness.";
            case "fps": return "Maximum displayed frames per second. Gameplay keeps its normal clock. Without smoothing, higher values repeat frames. With D3D12 native smoothing, choose your monitor refresh rate or Unlimited to remove the presentation cap.";
            case "interpolation": return "Experimental D3D12 motion interpolation between completed gameplay frames. Keeps normal game speed and adds about one game frame of delay. HUD, movies and unsupported surfaces retain their original frames. Can introduce artifacts and costs GPU/CPU time. Use 120 Hz or higher to benefit; Off is the default.";
            case "vsync": return "Synchronizes window redraws with your monitor to reduce tearing. May add latency and limits output to the monitor refresh rate.";
            case "camera": return "Modern orbit uses the right stick to orbit your monster when unlocked. Left-stick movement follows the camera: hold forward and turn the camera to steer. Target lock keeps the original target framing. Cutscenes and special cameras retain their original behavior. Both players have independent controls in split-screen. The camera-toggle action recenters the view. Original keeps the retail controls.";
            case "glyphs": return "Automatic selects face-button icons from the connected controller. Use Xbox or PlayStation to override detection. This changes prompts, not button bindings.";
            default: return "";
        }
    }
    void Defaults() { foreach (var pair in choices) pair.Value.SelectedItem = pair.Value.Tag; enabledMods.Clear(); }
    void LoadSettings()
    {
        try {
            if (!File.Exists(SettingsPath)) return;
            foreach (string line in File.ReadAllLines(SettingsPath)) {
                int split = line.IndexOf('='); if (split < 1) continue;
                ComboBox box; string value = line.Substring(split+1);
                if (line.Substring(0,split) == "renderer" && value == "Direct3D 12 (experimental)") value = "Direct3D 12";
                if (line.Substring(0,split) == "camera" && value == "Modern orbit (single-player)") value = "Modern orbit";
                if (line.Substring(0,split) == "mods") {
                    enabledMods.Clear();
                    foreach (string id in value.Split(new[] {','}, StringSplitOptions.RemoveEmptyEntries)) enabledMods.Add(id);
                    continue;
                }
                if (line.Substring(0,split) == "installed") { installed = value; continue; }
                if (line.Substring(0,split) == "iso") { iso.Text = value; continue; }
                if (choices.TryGetValue(line.Substring(0,split), out box) && box.Items.Contains(value)) box.SelectedItem = value;
            }
            if (enabledMods.Contains("goliath-prime")) enabledMods.Add("expanded-roster");
        } catch (IOException) { } catch (UnauthorizedAccessException) { }
    }
    void Save()
    {
        var lines = new List<string>();
        foreach (var pair in choices) lines.Add(pair.Key + "=" + pair.Value.Text);
        lines.Add("iso=" + iso.Text);
        lines.Add("installed=" + installed);
        lines.Add("mods=" + String.Join(",",enabledMods));
        Directory.CreateDirectory(Path.GetDirectoryName(SettingsPath));
        File.WriteAllLines(SettingsPath, lines);
    }
    string InstalledImage()
    {
        try {
            if (String.IsNullOrEmpty(installed)) return null;
            string path = Path.GetFullPath(Path.Combine(Root,installed));
            string parent = Path.Combine(Root,"game_data") + Path.DirectorySeparatorChar;
            if (!path.StartsWith(parent,StringComparison.OrdinalIgnoreCase)) return null;
            return GameImage.InstallationImage(path);
        } catch (InvalidDataException) { return null; } catch (IOException) { return null; }
        catch (UnauthorizedAccessException) { return null; } catch (ArgumentException) { return null; }
    }
    async void InstallData()
    {
        if (!File.Exists(iso.Text)) { Error(new IOException("Select your ISO with Browse first.")); return; }
        installing = true;
        foreach(Control c in Controls) c.Enabled=false;
        string source=iso.Text;
        try {
            GameExecutable(); // Detect an incomplete download before copying disc data.
            installed=await Task.Run(() => GameImage.Install(source,Root,message => BeginInvoke((Action)(() => status.Text=message))));
            installed=installed.Substring(Root.Length+1);
            Save(); RefreshArtwork(); status.Text="Installation complete. You can play without the original ISO.";
        } catch(Exception ex) { Error(ex); status.Text="Installation incomplete. Existing installations were kept."; }
        finally {
            installing=false; foreach(Control c in Controls) c.Enabled=true;
            choices["window"].Enabled=choices["mode"].Text!="Borderless";
            choices["interpolation"].Enabled=choices["renderer"].SelectedIndex==1;
        }
    }
    void Error(Exception ex) { MessageBox.Show(this, ex.Message, "Unable to start", MessageBoxButtons.OK, MessageBoxIcon.Error); }
    void Launch()
    {
        try {
            string cached = InstalledImage();
            if(!String.IsNullOrEmpty(installed) && cached==null)
                throw new IOException("Installed game data is missing or damaged. Select your ISO and choose Install Game again.");
            string image = cached ?? iso.Text;
            if (!File.Exists(image)) throw new IOException("Select your War of the Monsters ISO using Browse before playing.");
            ModSelection selectedMods = ModManager.Resolve(Path.Combine(Root,"mods"),enabledMods);
            if(cached!=null && Path.GetFileName(cached)=="disc.index")
                GameImage.PrepareInstalled(Path.GetDirectoryName(cached),Path.Combine(Root,"SCUS_971.97"));
            else GameImage.Prepare(image, Path.Combine(Root,"SCUS_971.97"));
            string exe = GameExecutable();
            Save();
            ProcessStartInfo start = new ProcessStartInfo(exe, "SCUS_971.97 \"" + image + "\"") {
                WorkingDirectory = Root, UseShellExecute = false, CreateNoWindow = true, RedirectStandardError = true };
            start.EnvironmentVariables.Remove("PS2X_MOD_PLAYER1");
            start.EnvironmentVariables.Remove("PS2X_MOD_DIAG");
            start.EnvironmentVariables.Remove("PS2X_MOD_ROSTER_IDS");
            start.EnvironmentVariables.Remove("PS2X_MOD_ROSTER_NAMES");
            start.EnvironmentVariables.Remove("PS2X_MOD_ROSTER_COSTUMES");
            start.EnvironmentVariables["PS2X_WOTM_SKIP_INTRO"] = selectedMods.SkipIntro ? "1" : "0";
            if (selectedMods.HasRoster) {
                string[] ids = new string[10], names = new string[10], costumes = new string[10];
                for (int slot=0; slot<10; ++slot) {
                    GameMod entry=selectedMods.Slots[slot];
                    ids[slot]=entry==null ? "0" : entry.GuestCharacter.ToString();
                    names[slot]=entry==null ? "" : RosterName(entry.Name);
                    costumes[slot]=entry==null ? "0" : entry.Costume.ToString();
                }
                start.EnvironmentVariables["PS2X_MOD_ROSTER_IDS"] = String.Join(",",ids);
                start.EnvironmentVariables["PS2X_MOD_ROSTER_NAMES"] = String.Join("|",names);
                start.EnvironmentVariables["PS2X_MOD_ROSTER_COSTUMES"] = String.Join(",",costumes);
            }
            start.EnvironmentVariables["PS2X_WINDOW"] = choices["window"].Text;
            start.EnvironmentVariables["PS2X_FULLSCREEN"] = choices["mode"].Text == "Fullscreen" ? "1" : "0";
            start.EnvironmentVariables["PS2X_BORDERLESS"] = choices["mode"].Text == "Borderless" ? "1" : "0";
            start.EnvironmentVariables["PS2X_GS_SCALE"] = choices["scale"].Text.Substring(0,1);
            start.EnvironmentVariables["PS2X_CAMERA"] = choices["camera"].SelectedIndex == 1 ? "modern" : "original";
            start.EnvironmentVariables["PS2X_ASPECT"] = choices["aspect"].SelectedIndex == 1 ? "16:9" : "original";
            bool d3d12 = choices["renderer"].SelectedIndex == 1;
            start.EnvironmentVariables["PS2X_GS_BACKEND"] = d3d12 ? "d3d12" : "opengl";
            start.EnvironmentVariables["PS2X_AA"] = choices["aa"].Text.ToLowerInvariant();
            start.EnvironmentVariables["PS2X_PRESENT_FPS"] = choices["fps"].Text == "Unlimited" ? "0" : choices["fps"].Text;
            bool smooth=choices["renderer"].SelectedIndex==1 && choices["interpolation"].SelectedIndex==1;
            start.EnvironmentVariables["PS2X_FRAME_INTERPOLATION"]=smooth?"1":"0";
            start.EnvironmentVariables["PS2X_DISPLAY_VSYNC"] = choices["vsync"].Text == "On" ? "1" : "0";
            start.EnvironmentVariables["PS2X_BUTTON_GLYPHS"] = choices["glyphs"].Text == "Automatic" ? "auto" : choices["glyphs"].Text.ToLowerInvariant();
            foreach (string key in new[] { "PS2X_VU1_FAST", "PS2X_NATIVE_PACING", "PS2X_GS_GPU" })
                if (String.IsNullOrEmpty(start.EnvironmentVariables[key])) start.EnvironmentVariables[key] = "1";
            rendererFailure = null;
            log = new StreamWriter(Path.Combine(Root, "wotm_last.log"), false); log.AutoFlush = true;
            game = new Process { StartInfo = start };
            game.ErrorDataReceived += delegate(object sender, DataReceivedEventArgs e) {
                if (e.Data != null) lock (logLock) {
                    if (e.Data.StartsWith("[gs:d3d12] ERROR:", StringComparison.Ordinal)) rendererFailure = e.Data;
                    if (log != null) log.WriteLine(e.Data);
                }
            };
            game.Start(); game.BeginErrorReadLine(); play.Enabled = false; Hide();
        } catch (Exception ex) {
            lock (logLock) { if (log != null) { log.Dispose(); log = null; } }
            if (game != null) { game.Dispose(); game = null; }
            Error(ex);
        }
    }
    static string RosterName(string value)
    {
        var letters=new System.Text.StringBuilder();
        foreach(char ch in value.ToUpperInvariant())
            if((ch>='A' && ch<='Z') || (ch>='0' && ch<='9') || ch==' ' || ch=='-') letters.Append(ch);
        return letters.Length>32 ? letters.ToString(0,32) : letters.ToString();
    }
    sealed class TimerHolder { public readonly System.Windows.Forms.Timer Timer = new System.Windows.Forms.Timer(); }
}
