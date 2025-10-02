#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.1.5),
    on April 01, 2025, at 14:51
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

import psychopy
psychopy.useVersion('2024.1.5')


# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'pygame'
prefs.hardware['audioLatencyMode'] = '1'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware, parallel
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.1.5'
expName = 'crea_freebreathing'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'UPN-ID': '999',
    'study-ID': '1417',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}


# ##########################################################
# ### get/read infos from "global parameter file" (gpar) ### 
# #####################################################
with open("../_settings/gpar") as myfile: 				#or 'open("../../_settings/gpar")'
    SessionData = [line.rstrip() for line in myfile]

# Data file name stem = absolute path + name;               | sess-admin |  | task | |   UPN-ID   |  |    ver    |   |  date/time |
#filename = _thisDir + os.sep + u'data/%s_%s_%s_run%s_%s' % (SessionData[1], expName, SessionData[0], SessionData[2], expInfo['date'])      #ver 2023.2.3
filename = _thisDir + os.sep + u'data/%s_%s_%s_run%s_%s' % (SessionData[1], expName, SessionData[0], SessionData[2], expInfo['date|hid'])   #ver 2024.1.5
##expInfo = {'UPN': SessionData[0], 'ver': SessionData[2]}    #add 'UPN-ID' and 'ver' to outputfile
##expInfo = {'admin': SessionData[1], 'UPN': SessionData[0], 'ver': SessionData[2]}    #add 'admin', 'UPN-ID' and 'ver' to outputfile
expInfo.update({'admin': SessionData[1], 'UPN': SessionData[0], 'ver': SessionData[2]})    #add 'admin', 'UPN-ID' and 'ver' to outputfile

# #####################################
# ### submit an ID for online-survey ### 
expInfo['UPN-ID'] = "%s%s%s%s%s" %(SessionData[1], '_', SessionData[0], '_run', SessionData[2])	# -> [1]=session_admin, [0]=UPN-ID, [2]=session_no
print("transmitted ID: %s" %expInfo['UPN-ID'])


# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1920, 1080]
_loggingLevel = logging.getLevel('error')
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # override logging level
    _loggingLevel = logging.getLevel(
        prefs.piloting['pilotLoggingLevel']
    )

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['UPN-ID'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\hdlabor\\Documents\\PsychoPy\\crea_breathing\\crea_freebreathing.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # this outputs to the screen, not a file
    logging.console.setLevel(_loggingLevel)
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log', level=_loggingLevel)
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowStencil=True,
            monitor='HD-Lab', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height', 
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    win.mouseVisible = False
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    ioSession = '1'
    if 'session' in expInfo:
        ioSession = str(expInfo['session'])
    ioServer = io.launchHubServer(window=win, **ioConfig)
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('m_ins_key') is None:
        # initialise m_ins_key
        m_ins_key = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='m_ins_key',
        )
    if deviceManager.getDevice('m_ins_key_1') is None:
        # initialise m_ins_key_1
        m_ins_key_1 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='m_ins_key_1',
        )
    if deviceManager.getDevice('m_ins_key_2') is None:
        # initialise m_ins_key_2
        m_ins_key_2 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='m_ins_key_2',
        )
    if deviceManager.getDevice('m_ins_key_3') is None:
        # initialise m_ins_key_3
        m_ins_key_3 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='m_ins_key_3',
        )
    if deviceManager.getDevice('nb_ins_key') is None:
        # initialise nb_ins_key
        nb_ins_key = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='nb_ins_key',
        )
    if deviceManager.getDevice('crea_tasks_key') is None:
        # initialise crea_tasks_key
        crea_tasks_key = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='crea_tasks_key',
        )
    if deviceManager.getDevice('AUT_idea_key') is None:
        # initialise AUT_idea_key
        AUT_idea_key = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='AUT_idea_key',
        )
    if deviceManager.getDevice('r_end_key') is None:
        # initialise r_end_key
        r_end_key = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='r_end_key',
        )
    if deviceManager.getDevice('TTCT_r_end_key') is None:
        # initialise TTCT_r_end_key
        TTCT_r_end_key = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='TTCT_r_end_key',
        )
    if deviceManager.getDevice('thx_key') is None:
        # initialise thx_key
        thx_key = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='thx_key',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # prevent components from auto-drawing
    win.stashAutoDraw()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # flip the screen
        win.flip()
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # restore auto-drawn components
    win.retrieveAutoDraw()
    # reset any timers
    for timer in timers:
        timer.reset()


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "_settings" ---
    # Run 'Begin Experiment' code from set_things
    ##  ##  ################################################################  ##  ##
    ## ##                                                                      ## ##
    ## ##                           CREA - BREATHING                           ## ##
    ## ##          AUT/TTCT * HYPEVENTILATION/PACED & NORMAL BREATHING         ## ##
    ## ##             (TTCT-digital on wacomONE (behavioral))                  ## ##
    ## ##                                                                      ## ##
    ## #########################|08/2024|bw (bernhard.weber@uni-graz.at)|CC-BY|## ##
    
    ##  import/load some extra modules/packages  ##
    import csv      #for stim-files
    import random   #for image sub-sample
    
    
    ##  Options for DEBUG/RESARCH-mode  ##
    DEBUG = 1   #0=research-mode; 1=debug-mode 
    if DEBUG:
        itemDur             = 18
        fixDur              = 0     #option: set time, use fixation
        AUTitem_dur         = 9
        descCrea_dur        = 3
        TTCTitem_dur        = 9
        TTCTdraft_time      = 15    #4debug  #Zeit zum Skizzieren >sketch + time2finish<
        time2finish         = 3     #4debug  #time 2 finish after warning/DING
        disp_thx            = 0
        
        breathing_dur       = 5     #duration natural/free breathing - starting condition
        crea_breathing_dur  = 3     #duration natural/free breathing - crea-items condition
    else:
        itemDur             = 180
        fixDur              = 0     #option: set time, use fixation
        AUTitem_dur         = 60
        descCrea_dur        = 10     #10s instruction/description of crea tasks
        TTCTitem_dur        = 60
        TTCTdraft_time      = 25    #4debug  #Zeit zum Skizzieren >sketch + time2finish<
        time2finish         = 3     #4debug  #time 2 finish after warning/DING
        disp_thx            = 1
        
        breathing_dur       = 120   #duration natural/free breathing (2min) - starting condition
        crea_breathing_dur  = 60    #duration natural/free breathing (1min) - crea-items condition
    
    ##  enable parallel port for markers  ##
    from psychopy import parallel
    Marker = parallel.ParallelPort(address=0x3FF8)
    
    
    ##  Options for ONSCREEN-KEYBOARD (touchscreen WACOM ONE)  ##
    ONSCREENKEYBOARD = 1   #0=NO onscreen keyboard; 1=ONSCREEN-KEYBOARD with touchscreen WACOM ONE
    if ONSCREENKEYBOARD:
        header_pos_y = .40
        main_pos_y   = .225
        btn_pos_y    = .05
        hint_pos_y   = 0
    else:
        header_pos_y = .40
        main_pos_y   = 0
        btn_pos_y    = -.25
        hint_pos_y   = -.45
    
    
    ##  initialize some vars  ##
    cwd = os.getcwd()
    win.mouseVisible = True
    warning = sound.Sound('stim//DING.ogg')
    
    NB_trials = 1   #natural breathing (cycle)s
    
    
    ##  search for window-handles by only a partial name  ##
    import win32gui
    import win32gui_struct
    
    def find_window_with_partial_name(partial_name):
        top_windows = []
    
        def _enum_windows_callback(hwnd, extra):
            nonlocal top_windows
            name = win32gui.GetWindowText(hwnd)
            if partial_name in name:
                top_windows.append((hwnd, name))
                
        win32gui_enumwindows_func = win32gui.EnumWindows
        win32gui_enumwindows_func(_enum_windows_callback, None)
    
        for hwnd, name in top_windows:
            if win32gui.IsWindowVisible(hwnd):
                return hwnd
    
        return None
    """
    # Example usage
    partial_window_name = "PartialName"
    window_handle = find_window_with_partial_name(partial_window_name)
    if window_handle:
        print(f"Window with partial name '{partial_window_name}' found, handle: {window_handle}")
    else:
        print(f"No window with partial name '{partial_window_name}' found")
    """
    
    
     ## ### # ###  # ## #  ## # ### ##
    ##  monsieur claude and it's ai  ##
    ###  ### ## # ### #  ##  ### # ## 
    
    ##  Function to read stimulus names from CSV file
    def read_stimuli_from_csv(filename, column_name='stimulus'):
        stimuli_from_file = []
        with open(filename, 'r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                stimuli_from_file.append(row[column_name])
        return stimuli_from_file
    
    
    ##  Function to create visual items: stimuli-names & prefixes, images, positions, sizes
    def create_items(stimuli_names, positions, prefix):
        items = []
        for stim_img, pos, i in zip(stimuli_names, positions, range(items2present)):
            item = visual.ImageStim(win, name=f'{prefix}_{i}', image=stim_img, pos=pos, size=(x_dim, y_dim))
            items.append(item)
        return items
    
    
    ##  Read AUT-stimuli (names) from CSV-file
    AUTitems_csv_filename = 'stim/AUTstim.csv'
    # ## stimuli_names = read_stimuli_from_csv(csv_filename, 'stimulus_name')   # Replace 'stimulus_name' with your actual column name
    AUT_items = read_stimuli_from_csv(AUTitems_csv_filename, 'AUT_items')
    #print(f'{AUT_items=}')
    
    
    ##  Read TTCTonline-stimuli (number(s)) from CSV-file
    TTCTitems_csv_filename = 'stim/TTCTonline_stim.csv'
    # ## stimuli_names = read_stimuli_from_csv(csv_filename, 'stimulus_name')   # Replace 'stimulus_name' with your actual column name
    TTCT_items = read_stimuli_from_csv(TTCTitems_csv_filename, 'TTCTonline_stim')
    #print(f'{TTCT_items=}')
    
    
    ##  Function to write a list to a .txt file
    def write_list_to_txt(filename, my_list):
        """
        Write a list to a text file, with each item on a new line.
        
        :param filename: Name of the text file to write to
        :param my_list: List of items to write
        """
        with open(filename, 'w') as file:
            for item in my_list:
                file.write(f"{item}\n")
    
    
    ##  Function to read a list from a .txt file
    def read_list_from_txt(filename):
        """
        Read a list from a text file, removing newline characters.
        
        :param filename: Name of the text file to read from
        :return: List of items read from the file
        """
        with open(filename, 'r') as file:
            return [line.strip() for line in file]
    
    # --- Initialize components for Routine "main_ins" ---
    m_ins_header = visual.TextStim(win=win, name='m_ins_header',
        text='Herzlich Willkommen zur Studie „Atme dich kreativ!“',
        font='Arial',
        pos=(0, .45), height=0.05, wrapWidth=1.25, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    m_ins_txt = visual.TextStim(win=win, name='m_ins_txt',
        text='Vielen Dank für Deine Bereitschaft, an dieser Untersuchung teilzunehmen. In dieser Studie untersuchen wir den Zusammenhang zwischen Alltagskreativität und der Herzratenvariabilität. \n\nDie Studie besteht aus insgesamt drei Blöcken zu je 20 Minuten. Zwischen den Blöcken gibt es jeweils eine 10-minütige Pause, in welcher zur Entspannung Naturbilder präsentiert werden. Pro Block wird 4-mal der Test zur verbalen und 4-mal der Test zur figuralen Kreativität vorgegeben. \n\nIn Block 1 wird noch keine Atemgeschwindigkeit vorgegeben, in Block 2 und 3 werden jene Atemgeschwindigkeiten anhand eines Atemkreises vorgegeben, welche du bereits aus den Übungsvideos kennst. \n\nWichtig: WÄHREND der Kreativitätstests atmest du in der Geschwindigkeit, DIE FÜR DICH PASSEND IST. Zwischen den Tests gibt es jeweils eine 1-minütige „Atemauffrischung“, in welcher wieder in der für den Block relevanten Geschwindigkeit geatmet wird.\n\nEs wird jeder Schritt im Laufe der Untersuchung noch einmal kurz erklärt. Falls es jetzt schon Fragen gibt, beantworte ich sie dir sehr gerne. Falls es keine Fragen gibt, drücke auf „Weiter“, um die Untersuchung zu starten.',
        font='Arial',
        pos=(0, 0), height=0.0325, wrapWidth=1.25, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    m_ins_key = keyboard.Keyboard(deviceName='m_ins_key')
    m_ins_btn = visual.Rect(
        win=win, name='m_ins_btn',
        width=(0.15, 0.05)[0], height=(0.15, 0.05)[1],
        ori=0.0, pos=(0, -.45), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='black', fillColor='lightgrey',
        opacity=None, depth=-4.0, interpolate=True)
    m_ins_hint = visual.TextStim(win=win, name='m_ins_hint',
        text='Weiter',
        font='Open Sans',
        pos=(0, -.45), height=0.025, wrapWidth=None, ori=0.0, 
        color='darkgreen', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    m_ins_mse = event.Mouse(win=win)
    x, y = [None, None]
    m_ins_mse.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "main_ins1" ---
    m_ins_header_1 = visual.TextStim(win=win, name='m_ins_header_1',
        text='Herzlich Willkommen zur Studie „Atme dich frei!“',
        font='Arial',
        pos=(0, .45), height=0.05, wrapWidth=1.25, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    m_ins_txt_1 = visual.TextStim(win=win, name='m_ins_txt_1',
        text='Vielen Dank für Deine Bereitschaft, an der Untersuchung teilzunehmen. \n\nDie Studie besteht aus insgesamt drei Blöcken zu je 20 Minuten. \n\nZwischen den Blöcken gibt es jeweils eine 5-minütige Pause, in welcher Naturbilder präsentiert werden. Pro Block wird 4-mal ein Task zur verbalen und 4-mal ein Task zur figuralen Kreativität vorgegeben.',
        font='Arial',
        pos=(0, main_pos_y), height=0.0325, wrapWidth=1.25, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    m_ins_key_1 = keyboard.Keyboard(deviceName='m_ins_key_1')
    m_ins_btn_1 = visual.Rect(
        win=win, name='m_ins_btn_1',
        width=(0.15, 0.05)[0], height=(0.15, 0.05)[1],
        ori=0.0, pos=(0, btn_pos_y), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='black', fillColor='lightgrey',
        opacity=None, depth=-4.0, interpolate=True)
    m_ins_hint_1 = visual.TextStim(win=win, name='m_ins_hint_1',
        text='Weiter',
        font='Open Sans',
        pos=(0, btn_pos_y), height=0.025, wrapWidth=None, ori=0.0, 
        color='darkgreen', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    m_ins_mse_1 = event.Mouse(win=win)
    x, y = [None, None]
    m_ins_mse_1.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "main_ins2" ---
    m_ins_header_2 = visual.TextStim(win=win, name='m_ins_header_2',
        text='Herzlich Willkommen zur Studie „Atme dich frei!“',
        font='Arial',
        pos=(0, .45), height=0.05, wrapWidth=1.25, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    m_ins_txt_2 = visual.TextStim(win=win, name='m_ins_txt_2',
        text='Im ersten Block gibt es keine festgelegte Atemgeschwindigkeit.\n\nIn den Blöcken 2 und 3 wird zu Beginn für zwei Minuten die Atemfrequenz durch einen Atemkreis vorgegeben, basierend auf den Rhythmen, die du bereits aus den Übungsvideos kennst.\n\nWichtig: WÄHREND der Kreativitätstasks atmest du in der Geschwindigkeit, die für dich ANGENEHM ist. Zwischen den Tasks wird wieder für 1 Minute geatmet.',
        font='Arial',
        pos=(0, main_pos_y), height=0.0325, wrapWidth=1.25, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    m_ins_key_2 = keyboard.Keyboard(deviceName='m_ins_key_2')
    m_ins_btn_2 = visual.Rect(
        win=win, name='m_ins_btn_2',
        width=(0.15, 0.05)[0], height=(0.15, 0.05)[1],
        ori=0.0, pos=(0, btn_pos_y), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='black', fillColor='lightgrey',
        opacity=None, depth=-4.0, interpolate=True)
    m_ins_hint_2 = visual.TextStim(win=win, name='m_ins_hint_2',
        text='Weiter',
        font='Open Sans',
        pos=(0, btn_pos_y), height=0.025, wrapWidth=None, ori=0.0, 
        color='darkgreen', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    m_ins_mse_2 = event.Mouse(win=win)
    x, y = [None, None]
    m_ins_mse_2.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "main_ins3" ---
    m_ins_header_3 = visual.TextStim(win=win, name='m_ins_header_3',
        text='Herzlich Willkommen zur Studie „Atme dich frei!“',
        font='Arial',
        pos=(0, .45), height=0.05, wrapWidth=1.25, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    m_ins_txt_3 = visual.TextStim(win=win, name='m_ins_txt_3',
        text='Es wird jeder Schritt im Laufe der Untersuchung noch einmal kurz erklärt.\n\nFalls es jetzt schon Fragen gibt, beantworte ich sie dir sehr gerne. Falls es keine Fragen gibt, drücke auf „Weiter“, um die Untersuchung zu starten.',
        font='Arial',
        pos=(0, main_pos_y), height=0.0325, wrapWidth=1.25, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    m_ins_key_3 = keyboard.Keyboard(deviceName='m_ins_key_3')
    m_ins_btn_3 = visual.Rect(
        win=win, name='m_ins_btn_3',
        width=(0.15, 0.05)[0], height=(0.15, 0.05)[1],
        ori=0.0, pos=(0, btn_pos_y), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='black', fillColor='lightgrey',
        opacity=None, depth=-4.0, interpolate=True)
    m_ins_hint_3 = visual.TextStim(win=win, name='m_ins_hint_3',
        text='Weiter',
        font='Open Sans',
        pos=(0, btn_pos_y), height=0.025, wrapWidth=None, ori=0.0, 
        color='darkgreen', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    m_ins_mse_3 = event.Mouse(win=win)
    x, y = [None, None]
    m_ins_mse_3.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "NB_ins" ---
    nb_ins_header = visual.TextStim(win=win, name='nb_ins_header',
        text='Spontanatmung',
        font='Arial',
        pos=(0, .4), height=0.05, wrapWidth=1.0, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    nb_ins_txt = visual.TextStim(win=win, name='nb_ins_txt',
        text='Atme für zwei Minuten in deiner individuellen Atemgeschwindigkeit,\nohne die Frequenz bewusst zu erhöhen oder zu verlangsamen. \n\nWenn du bereit bist > "Weiter"',
        font='Arial',
        pos=(0, main_pos_y), height=0.0325, wrapWidth=1.25, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    nb_ins_key = keyboard.Keyboard(deviceName='nb_ins_key')
    nb_ins_btn = visual.Rect(
        win=win, name='nb_ins_btn',
        width=(0.15, 0.05)[0], height=(0.15, 0.05)[1],
        ori=0.0, pos=(0, btn_pos_y), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='black', fillColor='lightgrey',
        opacity=None, depth=-4.0, interpolate=True)
    nb_ins_hint = visual.TextStim(win=win, name='nb_ins_hint',
        text='Weiter',
        font='Open Sans',
        pos=(0, btn_pos_y), height=0.025, wrapWidth=None, ori=0.0, 
        color='darkgreen', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    nb_ins_mse = event.Mouse(win=win)
    x, y = [None, None]
    nb_ins_mse.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "go_NB3min" ---
    startNB3min_txt = visual.TextStim(win=win, name='startNB3min_txt',
        text='...es beginnt das freie Atmen...',
        font='Open Sans',
        pos=(0, main_pos_y), height=0.035, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "NB_3min" ---
    NB3min_txt = visual.TextStim(win=win, name='NB3min_txt',
        text='...einfach ganz normal weiteratmen...',
        font='Open Sans',
        pos=(0, main_pos_y), height=0.035, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    NB3min_countdown = visual.TextStim(win=win, name='NB3min_countdown',
        text='',
        font='Open Sans',
        pos=(0, btn_pos_y), height=0.05, wrapWidth=None, ori=0.0, 
        color='blue', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "descCreaTasks" ---
    crea_tasks_header = visual.TextStim(win=win, name='crea_tasks_header',
        text='Beschreibung der Kreativitäts-Tasks (AUT und TTCT)',
        font='Arial',
        pos=(0, .4), height=0.05, wrapWidth=1.25, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    crea_tasks_txt = visual.TextStim(win=win, name='crea_tasks_txt',
        text='AUT = Worte von Alltagsgegenständen werden am Bildschirm gezeigt (z.B. "Buch", "Schirm", "Ziegelstein"). Nenne innerhalb von 60 Sekunden die originellste Verwendung, die dir für diesen Gegenstand einfällt!\n\nTTCT = Verschiedene Formen und unvollständige Figuren werden jeweils 60 Sekunden lang gezeigt. Bitte vervollständige dieses Bildfragment möglichst kreativ!',
        font='Arial',
        pos=(0, main_pos_y), height=0.0325, wrapWidth=1.25, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    crea_tasks_key = keyboard.Keyboard(deviceName='crea_tasks_key')
    crea_tasks_btn = visual.Rect(
        win=win, name='crea_tasks_btn',
        width=(0.15, 0.05)[0], height=(0.15, 0.05)[1],
        ori=0.0, pos=(0, btn_pos_y), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='black', fillColor='lightgrey',
        opacity=None, depth=-4.0, interpolate=True)
    crea_tasks_hint = visual.TextStim(win=win, name='crea_tasks_hint',
        text='Weiter',
        font='Open Sans',
        pos=(0, btn_pos_y), height=0.025, wrapWidth=None, ori=0.0, 
        color='darkgreen', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    crea_tasks_mse = event.Mouse(win=win)
    x, y = [None, None]
    crea_tasks_mse.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "crea_items_count" ---
    # Run 'Begin Experiment' code from cic_code
    go_NB1min = 1
    
    # --- Initialize components for Routine "AUT_info" ---
    
    # --- Initialize components for Routine "AUTtrial" ---
    # Run 'Begin Experiment' code from AUT_code
    presented_AUTitems = 0
    
    AUT_fix = visual.TextStim(win=win, name='AUT_fix',
        text='+',
        font='Open Sans',
        pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-1.0);
    AUT_mrk = parallel.ParallelPort(address='0x3FF8')
    AUT_item = visual.TextStim(win=win, name='AUT_item',
        text='',
        font='Arial',
        pos=(0, main_pos_y), height=0.05, wrapWidth=None, ori=0.0, 
        color='blue', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    AUT_idea_key = keyboard.Keyboard(deviceName='AUT_idea_key')
    AUT_idea_btn = visual.Rect(
        win=win, name='AUT_idea_btn',
        width=(0.125, 0.0475)[0], height=(0.125, 0.0475)[1],
        ori=0.0, pos=(0, btn_pos_y), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='darkgrey', fillColor='white',
        opacity=None, depth=-5.0, interpolate=True)
    AUT_idea_txt = visual.TextStim(win=win, name='AUT_idea_txt',
        text='Idee!',
        font='Arial',
        pos=(0, btn_pos_y), height=0.025, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    AUT_idea_mse = event.Mouse(win=win)
    x, y = [None, None]
    AUT_idea_mse.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "t_AUTanswer" ---
    ans_prompt = visual.TextStim(win=win, name='ans_prompt',
        text='Bitte tippen Sie hier Ihre (beste) Antwort ein:\n',
        font='Arial',
        pos=(0, header_pos_y), height=0.045, wrapWidth=1, ori=0, 
        color='black', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-1.0);
    answer_box = visual.TextBox2(
         win, text=None, placeholder=None, font='Arial',
         pos=(0, main_pos_y),     letterHeight=0.04,
         size=(0.75, 0.09), borderWidth=2.0,
         color='green', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='hidden',
         fillColor='lightgrey', borderColor='grey',
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=True,
         name='answer_box',
         depth=-2, autoLog=True,
    )
    go_on_btn = visual.Rect(
        win=win, name='go_on_btn',
        width=(0.125, 0.0475)[0], height=(0.125, 0.0475)[1],
        ori=0.0, pos=(0, btn_pos_y), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='darkgrey', fillColor='white',
        opacity=None, depth=-3.0, interpolate=True)
    go_on_txt = visual.TextStim(win=win, name='go_on_txt',
        text='Weiter',
        font='Arial',
        pos=(0, btn_pos_y), height=0.025, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    go_on_mse = event.Mouse(win=win)
    x, y = [None, None]
    go_on_mse.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "AUT_rating" ---
    r_header_txt = visual.TextStim(win=win, name='r_header_txt',
        text='Wie kreativ findest Du Deine Antwort',
        font='Arial',
        pos=(0, header_pos_y), height=0.045, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    VASrating = visual.Slider(win=win, name='VASrating',
        startValue=5, size=(1.0, 0.03), pos=(0, main_pos_y), units=win.units,
        labels=None, ticks=[0,1,2,3,4,5,6,7,8,9,10], granularity=0,
        style=['slider'], styleTweaks=(), opacity=1,
        labelColor='black', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.7035,
        flip=False, ori=0, depth=-2, readOnly=False)
    r_end_key = keyboard.Keyboard(deviceName='r_end_key')
    r1_mouse = event.Mouse(win=win)
    x, y = [None, None]
    r1_mouse.mouseClock = core.Clock()
    l_label = visual.TextStim(win=win, name='l_label',
        text='gar nicht kreativ',
        font='Arial',
        pos=(-.5, main_pos_y - .075), height=0.03, wrapWidth=None, ori=0.0, 
        color='blue', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    r_label = visual.TextStim(win=win, name='r_label',
        text='sehr kreativ',
        font='Arial',
        pos=(.5, main_pos_y - .075), height=0.03, wrapWidth=None, ori=0.0, 
        color='blue', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    r1_btn = visual.Rect(
        win=win, name='r1_btn',
        width=(0.125, 0.0475)[0], height=(0.125, 0.0475)[1],
        ori=0, pos=(0, btn_pos_y), anchor='center',
        lineWidth=1,     colorSpace='rgb',  lineColor=[1,1,1], fillColor=[1,1,1],
        opacity=1, depth=-7.0, interpolate=True)
    rating_hint = visual.TextStim(win=win, name='rating_hint',
        text='Weiter',
        font='Open Sans',
        pos=(0, btn_pos_y), height=0.025, wrapWidth=None, ori=0, 
        color='black', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-8.0);
    
    # --- Initialize components for Routine "b2TTCTonline" ---
    # Run 'Begin Experiment' code from b2_TTCT_code
    ## -------------------------------------------------------------------------- ##
    ##                          FOR BROWSER INTERACTION                           ##
    ## -------------------------------------------------------------------------- ##
    from selenium import webdriver
    #from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.common.action_chains import ActionChains
    from selenium.webdriver.chrome.options import Options   #>>disable "search-engine-choice" (07/2024)
    
    ##  claude script imports  ##
    #from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, NoAlertPresentException
    
    import win32gui
    import win32con
    
    ##  set/define some important variables
    presented_TTCTitems = 0
    #study_ID = '9999'              #default
    study_ID = '1417'               #!defined! study_id > 'Angerbauer' (2024/25) = study_ID = '1417'
    #study_ID = expInfo['study-ID']  #study_id 'Angerbauer' (2024/25)
    #participant_ID = '747'
    participant_ID = expInfo['UPN-ID']
    
    
    ##  ###   detect all the 'important' windows   ###  ##
    #if window_handle:
    #    print(f"Window with partial name '{partial_window_name}' found, handle: {window_handle}")
    #else:
    #    print(f"No window with partial name '{partial_window_name}' found")
    PsychoPyBuilderWin = find_window_with_partial_name("Builder")
    win32gui.ShowWindow(PsychoPyBuilderWin, win32con.SW_MINIMIZE)
    PsychoPyRunnerWin = find_window_with_partial_name("Runner")
    runParadigmaWindow = win32gui.FindWindow(None,"PsychoPy")
    
    
    ##  Initialize browser and other variables
    browser = None
    
    
    ##  Loading the webdriver by using the add-on "chromedriver.exe" in the root directory
    chrome_options = Options()                                              #>>disable "search-engine-choice" (07/2024)
    chrome_options.add_argument("--disable-search-engine-choice-screen")    #>>disable "search-engine-choice" (07/2024)
    browser = webdriver.Chrome(options=chrome_options)                      #chromedriver
    
    
    ##  def wait_for_alert_and_close(browser, timeout=60)  ##
    def wait_for_alert_and_close(browser, timeout=TTCTitem_dur):
        #Wait for alert to be present and then close browser when user clicks OK
        # :param driver: Selenium WebDriver instance
        # :param timeout: Maximum wait time for alert in seconds
        try:
            # ## Wait for alert to be present
            alert = WebDriverWait(browser, timeout).until(EC.alert_is_present())
            
            # ## Get alert text
            alert_text = alert.text
            #print(f"Alert text: {alert_text}")
            
            # ## Pause and wait for user to manually click OK
            #print("ALERT DETECTED: Please click OK on the alert.")
            #print("The script will wait for you to interact with the alert.")
            
            # ## Wait until alert is no longer present (indicating user clicked OK)
            WebDriverWait(browser, timeout).until_not(EC.alert_is_present())
            #print("TTCT item - done!")
        
        except NoAlertPresentException:
            print("No alert found.")
        except TimeoutException:
            print(f"No alert appeared or was not handled within {timeout} seconds.")
        finally:
            win32gui.ShowWindow(runParadigmaWindow, win32con.SW_MAXIMIZE)           #maximize "running paradigma"(-PsychoPy) window
            # ## minimize the browser window
            browser.minimize_window()                                               #minimize the browser window
            # ## Close and quit the browser
            #print("browser.close()")
            #browser.close()
            #print("browser.quit()")
            #browser.quit()
    
    
    
    # --- Initialize components for Routine "wait" ---
    
    # --- Initialize components for Routine "TTCT_rating" ---
    TTCT_r_header_txt = visual.TextStim(win=win, name='TTCT_r_header_txt',
        text='Wie kreativ findest Du Deine Antwort',
        font='Arial',
        pos=(0, header_pos_y), height=0.045, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    TTCT_VASrating = visual.Slider(win=win, name='TTCT_VASrating',
        startValue=5, size=(1.0, 0.03), pos=(0, main_pos_y), units=win.units,
        labels=None, ticks=[0,1,2,3,4,5,6,7,8,9,10], granularity=0,
        style=['slider'], styleTweaks=(), opacity=1,
        labelColor='black', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.7035,
        flip=False, ori=0, depth=-1, readOnly=False)
    TTCT_r_end_key = keyboard.Keyboard(deviceName='TTCT_r_end_key')
    TTCT_r1_mse = event.Mouse(win=win)
    x, y = [None, None]
    TTCT_r1_mse.mouseClock = core.Clock()
    TTCT_l_label = visual.TextStim(win=win, name='TTCT_l_label',
        text='gar nicht kreativ',
        font='Arial',
        pos=(-.5, main_pos_y - .075), height=0.03, wrapWidth=None, ori=0.0, 
        color='blue', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    TTCT_r_label = visual.TextStim(win=win, name='TTCT_r_label',
        text='sehr kreativ',
        font='Arial',
        pos=(.5, main_pos_y - .075), height=0.03, wrapWidth=None, ori=0.0, 
        color='blue', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    TTCT_r1_btn = visual.Rect(
        win=win, name='TTCT_r1_btn',
        width=(0.125, 0.0475)[0], height=(0.125, 0.0475)[1],
        ori=0, pos=(0, btn_pos_y), anchor='center',
        lineWidth=1,     colorSpace='rgb',  lineColor=[1,1,1], fillColor=[1,1,1],
        opacity=1, depth=-6.0, interpolate=True)
    TTCT_rating_hint = visual.TextStim(win=win, name='TTCT_rating_hint',
        text='Weiter',
        font='Open Sans',
        pos=(0, btn_pos_y), height=0.025, wrapWidth=None, ori=0, 
        color='black', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-7.0);
    
    # --- Initialize components for Routine "NB_1min" ---
    NB1min_txt = visual.TextStim(win=win, name='NB1min_txt',
        text='...einfach ganz normal weiteratmen...',
        font='Open Sans',
        pos=(0, main_pos_y), height=0.035, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    NB1m_countdown = visual.TextStim(win=win, name='NB1m_countdown',
        text='',
        font='Open Sans',
        pos=(0, btn_pos_y), height=0.075, wrapWidth=None, ori=0.0, 
        color='blue', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "thx" ---
    thx_txt = visual.TextStim(win=win, name='thx_txt',
        text='Vielen Dank,\n\ndieser Teil der Untersuchung ist zu Ende.',
        font='Open Sans',
        pos=(0, main_pos_y), height=0.035, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=0.0);
    thx_key = keyboard.Keyboard(deviceName='thx_key')
    thx_hint = visual.TextStim(win=win, name='thx_hint',
        text='Beenden mit Enter',
        font='Open Sans',
        pos=(.25, -.25), height=0.025, wrapWidth=None, ori=0, 
        color=[-0.25,-0.25,-0.25], colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-2.0);
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "_settings" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('_settings.started', globalClock.getTime(format='float'))
    # Run 'Begin Routine' code from set_things
    ##  marker: NB START  ##
    Marker.setData(201); core.wait(0.1); Marker.setData(0)
    
    ##  define some vars
    selected_actions_list = []          #mouse-click-selected actions
    selected_actionPictures_list = []   #image-names of selected actions
    selected_actions = 0                #selection counter
    
    actions_end_waiting = False         #wait to end time after last selection click
    
    
    ##  select (n) AUT-items for each block
    shuffle(AUT_items)          #aller guten 
    shuffle(AUT_items)          #Dinge sind
    shuffle(AUT_items)          #drei(mal)
    # AUTitems_block1 = random.sample(AUT_items, 4)           #items_block1 ... natural breathing
    AUTitems_block1 = AUT_items[:4]                 #items_block1 ... natural breathing
    #print(f'{AUTitems_block1=}')
    # AUTitems_block2 = random.sample(AUT_items, 4)           #items_block2 ... hyperventialation/SPB
    AUTitems_block2 = AUT_items[4:8]                #items_block2 ... hyperventialation/SPB
    #print(f'{AUTitems_block2=}')
    # AUTitems_block3 = random.sample(AUT_items, 4)           #items_block3 ... SPB/hyperventialation
    AUTitems_block3 = AUT_items[-4:]                 #items_block3 ... SPB/hyperventialation
    #print(f'{AUTitems_block3=}')
    
    ##  select (n) TTCT-items for each block
    shuffle(TTCT_items)          #aller guten 
    shuffle(TTCT_items)          #Dinge sind
    shuffle(TTCT_items)          #drei(mal)
    # TTCTitems_block1 = random.sample(TTCT_items, 4)           #items_block1 ... natural breathing
    TTCTitems_block1 = TTCT_items[:4]                 #items_block1 ... natural breathing
    #print(f'{TTCTitems_block1=}')
    # TTCTitems_block2 = random.sample(TTCT_items, 4)           #items_block2 ... hyperventialation/SPB
    TTCTitems_block2 = TTCT_items[4:8]                #items_block2 ... hyperventialation/SPB
    #print(f'{TTCTitems_block2=}')
    # TTCTitems_block3 = random.sample(TTCT_items, 4)           #items_block3 ... SPB/hyperventialation
    TTCTitems_block3 = TTCT_items[-4:]                 #items_block3 ... SPB/hyperventialation
    #print(f'{AUTitems_block3=}')
    
    
    ##  write the AUT lists to files
    #log_dir     = cwd + "//idea_log//" + expInfo['UPN-ID']
    write_list_to_txt(f'{cwd}//stim//subj_AUTlists//{participant_ID}_AUTitems_block1.txt', AUTitems_block1)
    write_list_to_txt(f'{cwd}//stim//subj_AUTlists//{participant_ID}_AUTitems_block2.txt', AUTitems_block2)
    write_list_to_txt(f'{cwd}//stim//subj_AUTlists//{participant_ID}_AUTitems_block3.txt', AUTitems_block3)
    
    ##  write the TTCT lists to files
    write_list_to_txt(f'{cwd}//stim//subj_TTCTlists//{participant_ID}_TTCTitems_block1.txt', TTCTitems_block1)
    write_list_to_txt(f'{cwd}//stim//subj_TTCTlists//{participant_ID}_TTCTitems_block2.txt', TTCTitems_block2)
    write_list_to_txt(f'{cwd}//stim//subj_TTCTlists//{participant_ID}_TTCTitems_block3.txt', TTCTitems_block3)
    
    
    ##  read (a) certain AUT list(s) from file(s)
    AUTitems_block1 = read_list_from_txt(f'{cwd}//stim//subj_AUTlists//{participant_ID}_AUTitems_block1.txt')
    #print(f'{AUTitems_block1=}')
    AUTitems_block2 = read_list_from_txt(f'{cwd}//stim//subj_AUTlists//{participant_ID}_AUTitems_block2.txt')
    #print(f'{AUTitems_block2=}')
    AUTitems_block3 = read_list_from_txt(f'{cwd}//stim//subj_AUTlists//{participant_ID}_AUTitems_block3.txt')
    #print(f'{AUTitems_block3=}')
    
    ##  read (a) certain TTCT list(s) from file(s)
    TTCTitems_block1 = read_list_from_txt(f'{cwd}//stim//subj_TTCTlists//{participant_ID}_TTCTitems_block1.txt')
    #print(f'{TTCTitems_block1=}')
    TTCTitems_block2 = read_list_from_txt(f'{cwd}//stim//subj_TTCTlists//{participant_ID}_TTCTitems_block2.txt')
    #print(f'{TTCTitems_block2=}')
    TTCTitems_block3 = read_list_from_txt(f'{cwd}//stim//subj_TTCTlists//{participant_ID}_TTCTitems_block3.txt')
    #print(f'{TTCTitems_block3=}')
    
    
    # keep track of which components have finished
    _settingsComponents = []
    for thisComponent in _settingsComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "_settings" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in _settingsComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "_settings" ---
    for thisComponent in _settingsComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('_settings.stopped', globalClock.getTime(format='float'))
    thisExp.nextEntry()
    # the Routine "_settings" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    main_ins_byp = data.TrialHandler(nReps=1.0, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='main_ins_byp')
    thisExp.addLoop(main_ins_byp)  # add the loop to the experiment
    thisMain_ins_byp = main_ins_byp.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisMain_ins_byp.rgb)
    if thisMain_ins_byp != None:
        for paramName in thisMain_ins_byp:
            globals()[paramName] = thisMain_ins_byp[paramName]
    
    for thisMain_ins_byp in main_ins_byp:
        currentLoop = main_ins_byp
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisMain_ins_byp.rgb)
        if thisMain_ins_byp != None:
            for paramName in thisMain_ins_byp:
                globals()[paramName] = thisMain_ins_byp[paramName]
        
        # set up handler to look after randomisation of conditions etc
        main_ins_orig = data.TrialHandler(nReps=0.0, method='sequential', 
            extraInfo=expInfo, originPath=-1,
            trialList=[None],
            seed=None, name='main_ins_orig')
        thisExp.addLoop(main_ins_orig)  # add the loop to the experiment
        thisMain_ins_orig = main_ins_orig.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisMain_ins_orig.rgb)
        if thisMain_ins_orig != None:
            for paramName in thisMain_ins_orig:
                globals()[paramName] = thisMain_ins_orig[paramName]
        
        for thisMain_ins_orig in main_ins_orig:
            currentLoop = main_ins_orig
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
            )
            # abbreviate parameter names if possible (e.g. rgb = thisMain_ins_orig.rgb)
            if thisMain_ins_orig != None:
                for paramName in thisMain_ins_orig:
                    globals()[paramName] = thisMain_ins_orig[paramName]
            
            # --- Prepare to start Routine "main_ins" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('main_ins.started', globalClock.getTime(format='float'))
            # Run 'Begin Routine' code from m_ins_code
            m_ins_txt.alignText = 'left'
            # create starting attributes for m_ins_key
            m_ins_key.keys = []
            m_ins_key.rt = []
            _m_ins_key_allKeys = []
            # setup some python lists for storing info about the m_ins_mse
            m_ins_mse.x = []
            m_ins_mse.y = []
            m_ins_mse.leftButton = []
            m_ins_mse.midButton = []
            m_ins_mse.rightButton = []
            m_ins_mse.time = []
            m_ins_mse.clicked_name = []
            gotValidClick = False  # until a click is received
            # keep track of which components have finished
            main_insComponents = [m_ins_header, m_ins_txt, m_ins_key, m_ins_btn, m_ins_hint, m_ins_mse]
            for thisComponent in main_insComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "main_ins" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *m_ins_header* updates
                
                # if m_ins_header is starting this frame...
                if m_ins_header.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    m_ins_header.frameNStart = frameN  # exact frame index
                    m_ins_header.tStart = t  # local t and not account for scr refresh
                    m_ins_header.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(m_ins_header, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'm_ins_header.started')
                    # update status
                    m_ins_header.status = STARTED
                    m_ins_header.setAutoDraw(True)
                
                # if m_ins_header is active this frame...
                if m_ins_header.status == STARTED:
                    # update params
                    pass
                
                # *m_ins_txt* updates
                
                # if m_ins_txt is starting this frame...
                if m_ins_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    m_ins_txt.frameNStart = frameN  # exact frame index
                    m_ins_txt.tStart = t  # local t and not account for scr refresh
                    m_ins_txt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(m_ins_txt, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'm_ins_txt.started')
                    # update status
                    m_ins_txt.status = STARTED
                    m_ins_txt.setAutoDraw(True)
                
                # if m_ins_txt is active this frame...
                if m_ins_txt.status == STARTED:
                    # update params
                    pass
                
                # *m_ins_key* updates
                waitOnFlip = False
                
                # if m_ins_key is starting this frame...
                if m_ins_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    m_ins_key.frameNStart = frameN  # exact frame index
                    m_ins_key.tStart = t  # local t and not account for scr refresh
                    m_ins_key.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(m_ins_key, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'm_ins_key.started')
                    # update status
                    m_ins_key.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(m_ins_key.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(m_ins_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if m_ins_key.status == STARTED and not waitOnFlip:
                    theseKeys = m_ins_key.getKeys(keyList=['return','space'], ignoreKeys=["escape"], waitRelease=False)
                    _m_ins_key_allKeys.extend(theseKeys)
                    if len(_m_ins_key_allKeys):
                        m_ins_key.keys = _m_ins_key_allKeys[-1].name  # just the last key pressed
                        m_ins_key.rt = _m_ins_key_allKeys[-1].rt
                        m_ins_key.duration = _m_ins_key_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # *m_ins_btn* updates
                
                # if m_ins_btn is starting this frame...
                if m_ins_btn.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    m_ins_btn.frameNStart = frameN  # exact frame index
                    m_ins_btn.tStart = t  # local t and not account for scr refresh
                    m_ins_btn.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(m_ins_btn, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'm_ins_btn.started')
                    # update status
                    m_ins_btn.status = STARTED
                    m_ins_btn.setAutoDraw(True)
                
                # if m_ins_btn is active this frame...
                if m_ins_btn.status == STARTED:
                    # update params
                    pass
                
                # *m_ins_hint* updates
                
                # if m_ins_hint is starting this frame...
                if m_ins_hint.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    m_ins_hint.frameNStart = frameN  # exact frame index
                    m_ins_hint.tStart = t  # local t and not account for scr refresh
                    m_ins_hint.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(m_ins_hint, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'm_ins_hint.started')
                    # update status
                    m_ins_hint.status = STARTED
                    m_ins_hint.setAutoDraw(True)
                
                # if m_ins_hint is active this frame...
                if m_ins_hint.status == STARTED:
                    # update params
                    pass
                # *m_ins_mse* updates
                
                # if m_ins_mse is starting this frame...
                if m_ins_mse.status == NOT_STARTED and t >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    m_ins_mse.frameNStart = frameN  # exact frame index
                    m_ins_mse.tStart = t  # local t and not account for scr refresh
                    m_ins_mse.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(m_ins_mse, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.addData('m_ins_mse.started', t)
                    # update status
                    m_ins_mse.status = STARTED
                    m_ins_mse.mouseClock.reset()
                    prevButtonState = m_ins_mse.getPressed()  # if button is down already this ISN'T a new click
                if m_ins_mse.status == STARTED:  # only update if started and not finished!
                    buttons = m_ins_mse.getPressed()
                    if buttons != prevButtonState:  # button state changed?
                        prevButtonState = buttons
                        if sum(buttons) > 0:  # state changed to a new click
                            # check if the mouse was inside our 'clickable' objects
                            gotValidClick = False
                            clickableList = environmenttools.getFromNames(m_ins_btn, namespace=locals())
                            for obj in clickableList:
                                # is this object clicked on?
                                if obj.contains(m_ins_mse):
                                    gotValidClick = True
                                    m_ins_mse.clicked_name.append(obj.name)
                            x, y = m_ins_mse.getPos()
                            m_ins_mse.x.append(x)
                            m_ins_mse.y.append(y)
                            buttons = m_ins_mse.getPressed()
                            m_ins_mse.leftButton.append(buttons[0])
                            m_ins_mse.midButton.append(buttons[1])
                            m_ins_mse.rightButton.append(buttons[2])
                            m_ins_mse.time.append(m_ins_mse.mouseClock.getTime())
                            if gotValidClick:
                                continueRoutine = False  # end routine on response
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in main_insComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "main_ins" ---
            for thisComponent in main_insComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('main_ins.stopped', globalClock.getTime(format='float'))
            # check responses
            if m_ins_key.keys in ['', [], None]:  # No response was made
                m_ins_key.keys = None
            main_ins_orig.addData('m_ins_key.keys',m_ins_key.keys)
            if m_ins_key.keys != None:  # we had a response
                main_ins_orig.addData('m_ins_key.rt', m_ins_key.rt)
                main_ins_orig.addData('m_ins_key.duration', m_ins_key.duration)
            # store data for main_ins_orig (TrialHandler)
            main_ins_orig.addData('m_ins_mse.x', m_ins_mse.x)
            main_ins_orig.addData('m_ins_mse.y', m_ins_mse.y)
            main_ins_orig.addData('m_ins_mse.leftButton', m_ins_mse.leftButton)
            main_ins_orig.addData('m_ins_mse.midButton', m_ins_mse.midButton)
            main_ins_orig.addData('m_ins_mse.rightButton', m_ins_mse.rightButton)
            main_ins_orig.addData('m_ins_mse.time', m_ins_mse.time)
            main_ins_orig.addData('m_ins_mse.clicked_name', m_ins_mse.clicked_name)
            # the Routine "main_ins" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
        # completed 0.0 repeats of 'main_ins_orig'
        
        
        # --- Prepare to start Routine "main_ins1" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('main_ins1.started', globalClock.getTime(format='float'))
        # Run 'Begin Routine' code from m_ins_code_1
        m_ins_txt_1.alignText = 'left'
        # create starting attributes for m_ins_key_1
        m_ins_key_1.keys = []
        m_ins_key_1.rt = []
        _m_ins_key_1_allKeys = []
        # setup some python lists for storing info about the m_ins_mse_1
        m_ins_mse_1.x = []
        m_ins_mse_1.y = []
        m_ins_mse_1.leftButton = []
        m_ins_mse_1.midButton = []
        m_ins_mse_1.rightButton = []
        m_ins_mse_1.time = []
        m_ins_mse_1.clicked_name = []
        gotValidClick = False  # until a click is received
        # keep track of which components have finished
        main_ins1Components = [m_ins_header_1, m_ins_txt_1, m_ins_key_1, m_ins_btn_1, m_ins_hint_1, m_ins_mse_1]
        for thisComponent in main_ins1Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "main_ins1" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *m_ins_header_1* updates
            
            # if m_ins_header_1 is starting this frame...
            if m_ins_header_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                m_ins_header_1.frameNStart = frameN  # exact frame index
                m_ins_header_1.tStart = t  # local t and not account for scr refresh
                m_ins_header_1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(m_ins_header_1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'm_ins_header_1.started')
                # update status
                m_ins_header_1.status = STARTED
                m_ins_header_1.setAutoDraw(True)
            
            # if m_ins_header_1 is active this frame...
            if m_ins_header_1.status == STARTED:
                # update params
                pass
            
            # *m_ins_txt_1* updates
            
            # if m_ins_txt_1 is starting this frame...
            if m_ins_txt_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                m_ins_txt_1.frameNStart = frameN  # exact frame index
                m_ins_txt_1.tStart = t  # local t and not account for scr refresh
                m_ins_txt_1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(m_ins_txt_1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'm_ins_txt_1.started')
                # update status
                m_ins_txt_1.status = STARTED
                m_ins_txt_1.setAutoDraw(True)
            
            # if m_ins_txt_1 is active this frame...
            if m_ins_txt_1.status == STARTED:
                # update params
                pass
            
            # *m_ins_key_1* updates
            waitOnFlip = False
            
            # if m_ins_key_1 is starting this frame...
            if m_ins_key_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                m_ins_key_1.frameNStart = frameN  # exact frame index
                m_ins_key_1.tStart = t  # local t and not account for scr refresh
                m_ins_key_1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(m_ins_key_1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'm_ins_key_1.started')
                # update status
                m_ins_key_1.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(m_ins_key_1.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(m_ins_key_1.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if m_ins_key_1.status == STARTED and not waitOnFlip:
                theseKeys = m_ins_key_1.getKeys(keyList=['return','space'], ignoreKeys=["escape"], waitRelease=False)
                _m_ins_key_1_allKeys.extend(theseKeys)
                if len(_m_ins_key_1_allKeys):
                    m_ins_key_1.keys = _m_ins_key_1_allKeys[-1].name  # just the last key pressed
                    m_ins_key_1.rt = _m_ins_key_1_allKeys[-1].rt
                    m_ins_key_1.duration = _m_ins_key_1_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *m_ins_btn_1* updates
            
            # if m_ins_btn_1 is starting this frame...
            if m_ins_btn_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                m_ins_btn_1.frameNStart = frameN  # exact frame index
                m_ins_btn_1.tStart = t  # local t and not account for scr refresh
                m_ins_btn_1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(m_ins_btn_1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'm_ins_btn_1.started')
                # update status
                m_ins_btn_1.status = STARTED
                m_ins_btn_1.setAutoDraw(True)
            
            # if m_ins_btn_1 is active this frame...
            if m_ins_btn_1.status == STARTED:
                # update params
                pass
            
            # *m_ins_hint_1* updates
            
            # if m_ins_hint_1 is starting this frame...
            if m_ins_hint_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                m_ins_hint_1.frameNStart = frameN  # exact frame index
                m_ins_hint_1.tStart = t  # local t and not account for scr refresh
                m_ins_hint_1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(m_ins_hint_1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'm_ins_hint_1.started')
                # update status
                m_ins_hint_1.status = STARTED
                m_ins_hint_1.setAutoDraw(True)
            
            # if m_ins_hint_1 is active this frame...
            if m_ins_hint_1.status == STARTED:
                # update params
                pass
            # *m_ins_mse_1* updates
            
            # if m_ins_mse_1 is starting this frame...
            if m_ins_mse_1.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                m_ins_mse_1.frameNStart = frameN  # exact frame index
                m_ins_mse_1.tStart = t  # local t and not account for scr refresh
                m_ins_mse_1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(m_ins_mse_1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('m_ins_mse_1.started', t)
                # update status
                m_ins_mse_1.status = STARTED
                m_ins_mse_1.mouseClock.reset()
                prevButtonState = m_ins_mse_1.getPressed()  # if button is down already this ISN'T a new click
            if m_ins_mse_1.status == STARTED:  # only update if started and not finished!
                buttons = m_ins_mse_1.getPressed()
                if buttons != prevButtonState:  # button state changed?
                    prevButtonState = buttons
                    if sum(buttons) > 0:  # state changed to a new click
                        # check if the mouse was inside our 'clickable' objects
                        gotValidClick = False
                        clickableList = environmenttools.getFromNames(m_ins_btn_1, namespace=locals())
                        for obj in clickableList:
                            # is this object clicked on?
                            if obj.contains(m_ins_mse_1):
                                gotValidClick = True
                                m_ins_mse_1.clicked_name.append(obj.name)
                        x, y = m_ins_mse_1.getPos()
                        m_ins_mse_1.x.append(x)
                        m_ins_mse_1.y.append(y)
                        buttons = m_ins_mse_1.getPressed()
                        m_ins_mse_1.leftButton.append(buttons[0])
                        m_ins_mse_1.midButton.append(buttons[1])
                        m_ins_mse_1.rightButton.append(buttons[2])
                        m_ins_mse_1.time.append(m_ins_mse_1.mouseClock.getTime())
                        if gotValidClick:
                            continueRoutine = False  # end routine on response
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in main_ins1Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "main_ins1" ---
        for thisComponent in main_ins1Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('main_ins1.stopped', globalClock.getTime(format='float'))
        # check responses
        if m_ins_key_1.keys in ['', [], None]:  # No response was made
            m_ins_key_1.keys = None
        main_ins_byp.addData('m_ins_key_1.keys',m_ins_key_1.keys)
        if m_ins_key_1.keys != None:  # we had a response
            main_ins_byp.addData('m_ins_key_1.rt', m_ins_key_1.rt)
            main_ins_byp.addData('m_ins_key_1.duration', m_ins_key_1.duration)
        # store data for main_ins_byp (TrialHandler)
        main_ins_byp.addData('m_ins_mse_1.x', m_ins_mse_1.x)
        main_ins_byp.addData('m_ins_mse_1.y', m_ins_mse_1.y)
        main_ins_byp.addData('m_ins_mse_1.leftButton', m_ins_mse_1.leftButton)
        main_ins_byp.addData('m_ins_mse_1.midButton', m_ins_mse_1.midButton)
        main_ins_byp.addData('m_ins_mse_1.rightButton', m_ins_mse_1.rightButton)
        main_ins_byp.addData('m_ins_mse_1.time', m_ins_mse_1.time)
        main_ins_byp.addData('m_ins_mse_1.clicked_name', m_ins_mse_1.clicked_name)
        # the Routine "main_ins1" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "main_ins2" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('main_ins2.started', globalClock.getTime(format='float'))
        # Run 'Begin Routine' code from m_ins_code_2
        m_ins_txt_2.alignText = 'left'
        # create starting attributes for m_ins_key_2
        m_ins_key_2.keys = []
        m_ins_key_2.rt = []
        _m_ins_key_2_allKeys = []
        # setup some python lists for storing info about the m_ins_mse_2
        m_ins_mse_2.x = []
        m_ins_mse_2.y = []
        m_ins_mse_2.leftButton = []
        m_ins_mse_2.midButton = []
        m_ins_mse_2.rightButton = []
        m_ins_mse_2.time = []
        m_ins_mse_2.clicked_name = []
        gotValidClick = False  # until a click is received
        # keep track of which components have finished
        main_ins2Components = [m_ins_header_2, m_ins_txt_2, m_ins_key_2, m_ins_btn_2, m_ins_hint_2, m_ins_mse_2]
        for thisComponent in main_ins2Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "main_ins2" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *m_ins_header_2* updates
            
            # if m_ins_header_2 is starting this frame...
            if m_ins_header_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                m_ins_header_2.frameNStart = frameN  # exact frame index
                m_ins_header_2.tStart = t  # local t and not account for scr refresh
                m_ins_header_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(m_ins_header_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'm_ins_header_2.started')
                # update status
                m_ins_header_2.status = STARTED
                m_ins_header_2.setAutoDraw(True)
            
            # if m_ins_header_2 is active this frame...
            if m_ins_header_2.status == STARTED:
                # update params
                pass
            
            # *m_ins_txt_2* updates
            
            # if m_ins_txt_2 is starting this frame...
            if m_ins_txt_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                m_ins_txt_2.frameNStart = frameN  # exact frame index
                m_ins_txt_2.tStart = t  # local t and not account for scr refresh
                m_ins_txt_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(m_ins_txt_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'm_ins_txt_2.started')
                # update status
                m_ins_txt_2.status = STARTED
                m_ins_txt_2.setAutoDraw(True)
            
            # if m_ins_txt_2 is active this frame...
            if m_ins_txt_2.status == STARTED:
                # update params
                pass
            
            # *m_ins_key_2* updates
            waitOnFlip = False
            
            # if m_ins_key_2 is starting this frame...
            if m_ins_key_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                m_ins_key_2.frameNStart = frameN  # exact frame index
                m_ins_key_2.tStart = t  # local t and not account for scr refresh
                m_ins_key_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(m_ins_key_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'm_ins_key_2.started')
                # update status
                m_ins_key_2.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(m_ins_key_2.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(m_ins_key_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if m_ins_key_2.status == STARTED and not waitOnFlip:
                theseKeys = m_ins_key_2.getKeys(keyList=['return','space'], ignoreKeys=["escape"], waitRelease=False)
                _m_ins_key_2_allKeys.extend(theseKeys)
                if len(_m_ins_key_2_allKeys):
                    m_ins_key_2.keys = _m_ins_key_2_allKeys[-1].name  # just the last key pressed
                    m_ins_key_2.rt = _m_ins_key_2_allKeys[-1].rt
                    m_ins_key_2.duration = _m_ins_key_2_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *m_ins_btn_2* updates
            
            # if m_ins_btn_2 is starting this frame...
            if m_ins_btn_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                m_ins_btn_2.frameNStart = frameN  # exact frame index
                m_ins_btn_2.tStart = t  # local t and not account for scr refresh
                m_ins_btn_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(m_ins_btn_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'm_ins_btn_2.started')
                # update status
                m_ins_btn_2.status = STARTED
                m_ins_btn_2.setAutoDraw(True)
            
            # if m_ins_btn_2 is active this frame...
            if m_ins_btn_2.status == STARTED:
                # update params
                pass
            
            # *m_ins_hint_2* updates
            
            # if m_ins_hint_2 is starting this frame...
            if m_ins_hint_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                m_ins_hint_2.frameNStart = frameN  # exact frame index
                m_ins_hint_2.tStart = t  # local t and not account for scr refresh
                m_ins_hint_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(m_ins_hint_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'm_ins_hint_2.started')
                # update status
                m_ins_hint_2.status = STARTED
                m_ins_hint_2.setAutoDraw(True)
            
            # if m_ins_hint_2 is active this frame...
            if m_ins_hint_2.status == STARTED:
                # update params
                pass
            # *m_ins_mse_2* updates
            
            # if m_ins_mse_2 is starting this frame...
            if m_ins_mse_2.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                m_ins_mse_2.frameNStart = frameN  # exact frame index
                m_ins_mse_2.tStart = t  # local t and not account for scr refresh
                m_ins_mse_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(m_ins_mse_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('m_ins_mse_2.started', t)
                # update status
                m_ins_mse_2.status = STARTED
                m_ins_mse_2.mouseClock.reset()
                prevButtonState = m_ins_mse_2.getPressed()  # if button is down already this ISN'T a new click
            if m_ins_mse_2.status == STARTED:  # only update if started and not finished!
                buttons = m_ins_mse_2.getPressed()
                if buttons != prevButtonState:  # button state changed?
                    prevButtonState = buttons
                    if sum(buttons) > 0:  # state changed to a new click
                        # check if the mouse was inside our 'clickable' objects
                        gotValidClick = False
                        clickableList = environmenttools.getFromNames(m_ins_btn_2, namespace=locals())
                        for obj in clickableList:
                            # is this object clicked on?
                            if obj.contains(m_ins_mse_2):
                                gotValidClick = True
                                m_ins_mse_2.clicked_name.append(obj.name)
                        x, y = m_ins_mse_2.getPos()
                        m_ins_mse_2.x.append(x)
                        m_ins_mse_2.y.append(y)
                        buttons = m_ins_mse_2.getPressed()
                        m_ins_mse_2.leftButton.append(buttons[0])
                        m_ins_mse_2.midButton.append(buttons[1])
                        m_ins_mse_2.rightButton.append(buttons[2])
                        m_ins_mse_2.time.append(m_ins_mse_2.mouseClock.getTime())
                        if gotValidClick:
                            continueRoutine = False  # end routine on response
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in main_ins2Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "main_ins2" ---
        for thisComponent in main_ins2Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('main_ins2.stopped', globalClock.getTime(format='float'))
        # check responses
        if m_ins_key_2.keys in ['', [], None]:  # No response was made
            m_ins_key_2.keys = None
        main_ins_byp.addData('m_ins_key_2.keys',m_ins_key_2.keys)
        if m_ins_key_2.keys != None:  # we had a response
            main_ins_byp.addData('m_ins_key_2.rt', m_ins_key_2.rt)
            main_ins_byp.addData('m_ins_key_2.duration', m_ins_key_2.duration)
        # store data for main_ins_byp (TrialHandler)
        main_ins_byp.addData('m_ins_mse_2.x', m_ins_mse_2.x)
        main_ins_byp.addData('m_ins_mse_2.y', m_ins_mse_2.y)
        main_ins_byp.addData('m_ins_mse_2.leftButton', m_ins_mse_2.leftButton)
        main_ins_byp.addData('m_ins_mse_2.midButton', m_ins_mse_2.midButton)
        main_ins_byp.addData('m_ins_mse_2.rightButton', m_ins_mse_2.rightButton)
        main_ins_byp.addData('m_ins_mse_2.time', m_ins_mse_2.time)
        main_ins_byp.addData('m_ins_mse_2.clicked_name', m_ins_mse_2.clicked_name)
        # the Routine "main_ins2" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "main_ins3" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('main_ins3.started', globalClock.getTime(format='float'))
        # Run 'Begin Routine' code from m_ins_code_3
        m_ins_txt_3.alignText = 'left'
        # create starting attributes for m_ins_key_3
        m_ins_key_3.keys = []
        m_ins_key_3.rt = []
        _m_ins_key_3_allKeys = []
        # setup some python lists for storing info about the m_ins_mse_3
        m_ins_mse_3.x = []
        m_ins_mse_3.y = []
        m_ins_mse_3.leftButton = []
        m_ins_mse_3.midButton = []
        m_ins_mse_3.rightButton = []
        m_ins_mse_3.time = []
        m_ins_mse_3.clicked_name = []
        gotValidClick = False  # until a click is received
        # keep track of which components have finished
        main_ins3Components = [m_ins_header_3, m_ins_txt_3, m_ins_key_3, m_ins_btn_3, m_ins_hint_3, m_ins_mse_3]
        for thisComponent in main_ins3Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "main_ins3" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *m_ins_header_3* updates
            
            # if m_ins_header_3 is starting this frame...
            if m_ins_header_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                m_ins_header_3.frameNStart = frameN  # exact frame index
                m_ins_header_3.tStart = t  # local t and not account for scr refresh
                m_ins_header_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(m_ins_header_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'm_ins_header_3.started')
                # update status
                m_ins_header_3.status = STARTED
                m_ins_header_3.setAutoDraw(True)
            
            # if m_ins_header_3 is active this frame...
            if m_ins_header_3.status == STARTED:
                # update params
                pass
            
            # *m_ins_txt_3* updates
            
            # if m_ins_txt_3 is starting this frame...
            if m_ins_txt_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                m_ins_txt_3.frameNStart = frameN  # exact frame index
                m_ins_txt_3.tStart = t  # local t and not account for scr refresh
                m_ins_txt_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(m_ins_txt_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'm_ins_txt_3.started')
                # update status
                m_ins_txt_3.status = STARTED
                m_ins_txt_3.setAutoDraw(True)
            
            # if m_ins_txt_3 is active this frame...
            if m_ins_txt_3.status == STARTED:
                # update params
                pass
            
            # *m_ins_key_3* updates
            waitOnFlip = False
            
            # if m_ins_key_3 is starting this frame...
            if m_ins_key_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                m_ins_key_3.frameNStart = frameN  # exact frame index
                m_ins_key_3.tStart = t  # local t and not account for scr refresh
                m_ins_key_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(m_ins_key_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'm_ins_key_3.started')
                # update status
                m_ins_key_3.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(m_ins_key_3.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(m_ins_key_3.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if m_ins_key_3.status == STARTED and not waitOnFlip:
                theseKeys = m_ins_key_3.getKeys(keyList=['return','space'], ignoreKeys=["escape"], waitRelease=False)
                _m_ins_key_3_allKeys.extend(theseKeys)
                if len(_m_ins_key_3_allKeys):
                    m_ins_key_3.keys = _m_ins_key_3_allKeys[-1].name  # just the last key pressed
                    m_ins_key_3.rt = _m_ins_key_3_allKeys[-1].rt
                    m_ins_key_3.duration = _m_ins_key_3_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *m_ins_btn_3* updates
            
            # if m_ins_btn_3 is starting this frame...
            if m_ins_btn_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                m_ins_btn_3.frameNStart = frameN  # exact frame index
                m_ins_btn_3.tStart = t  # local t and not account for scr refresh
                m_ins_btn_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(m_ins_btn_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'm_ins_btn_3.started')
                # update status
                m_ins_btn_3.status = STARTED
                m_ins_btn_3.setAutoDraw(True)
            
            # if m_ins_btn_3 is active this frame...
            if m_ins_btn_3.status == STARTED:
                # update params
                pass
            
            # *m_ins_hint_3* updates
            
            # if m_ins_hint_3 is starting this frame...
            if m_ins_hint_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                m_ins_hint_3.frameNStart = frameN  # exact frame index
                m_ins_hint_3.tStart = t  # local t and not account for scr refresh
                m_ins_hint_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(m_ins_hint_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'm_ins_hint_3.started')
                # update status
                m_ins_hint_3.status = STARTED
                m_ins_hint_3.setAutoDraw(True)
            
            # if m_ins_hint_3 is active this frame...
            if m_ins_hint_3.status == STARTED:
                # update params
                pass
            # *m_ins_mse_3* updates
            
            # if m_ins_mse_3 is starting this frame...
            if m_ins_mse_3.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                m_ins_mse_3.frameNStart = frameN  # exact frame index
                m_ins_mse_3.tStart = t  # local t and not account for scr refresh
                m_ins_mse_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(m_ins_mse_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('m_ins_mse_3.started', t)
                # update status
                m_ins_mse_3.status = STARTED
                m_ins_mse_3.mouseClock.reset()
                prevButtonState = m_ins_mse_3.getPressed()  # if button is down already this ISN'T a new click
            if m_ins_mse_3.status == STARTED:  # only update if started and not finished!
                buttons = m_ins_mse_3.getPressed()
                if buttons != prevButtonState:  # button state changed?
                    prevButtonState = buttons
                    if sum(buttons) > 0:  # state changed to a new click
                        # check if the mouse was inside our 'clickable' objects
                        gotValidClick = False
                        clickableList = environmenttools.getFromNames(m_ins_btn_3, namespace=locals())
                        for obj in clickableList:
                            # is this object clicked on?
                            if obj.contains(m_ins_mse_3):
                                gotValidClick = True
                                m_ins_mse_3.clicked_name.append(obj.name)
                        x, y = m_ins_mse_3.getPos()
                        m_ins_mse_3.x.append(x)
                        m_ins_mse_3.y.append(y)
                        buttons = m_ins_mse_3.getPressed()
                        m_ins_mse_3.leftButton.append(buttons[0])
                        m_ins_mse_3.midButton.append(buttons[1])
                        m_ins_mse_3.rightButton.append(buttons[2])
                        m_ins_mse_3.time.append(m_ins_mse_3.mouseClock.getTime())
                        if gotValidClick:
                            continueRoutine = False  # end routine on response
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in main_ins3Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "main_ins3" ---
        for thisComponent in main_ins3Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('main_ins3.stopped', globalClock.getTime(format='float'))
        # check responses
        if m_ins_key_3.keys in ['', [], None]:  # No response was made
            m_ins_key_3.keys = None
        main_ins_byp.addData('m_ins_key_3.keys',m_ins_key_3.keys)
        if m_ins_key_3.keys != None:  # we had a response
            main_ins_byp.addData('m_ins_key_3.rt', m_ins_key_3.rt)
            main_ins_byp.addData('m_ins_key_3.duration', m_ins_key_3.duration)
        # store data for main_ins_byp (TrialHandler)
        main_ins_byp.addData('m_ins_mse_3.x', m_ins_mse_3.x)
        main_ins_byp.addData('m_ins_mse_3.y', m_ins_mse_3.y)
        main_ins_byp.addData('m_ins_mse_3.leftButton', m_ins_mse_3.leftButton)
        main_ins_byp.addData('m_ins_mse_3.midButton', m_ins_mse_3.midButton)
        main_ins_byp.addData('m_ins_mse_3.rightButton', m_ins_mse_3.rightButton)
        main_ins_byp.addData('m_ins_mse_3.time', m_ins_mse_3.time)
        main_ins_byp.addData('m_ins_mse_3.clicked_name', m_ins_mse_3.clicked_name)
        # the Routine "main_ins3" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
    # completed 1.0 repeats of 'main_ins_byp'
    
    
    # set up handler to look after randomisation of conditions etc
    NB_ins_byp = data.TrialHandler(nReps=1.0, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='NB_ins_byp')
    thisExp.addLoop(NB_ins_byp)  # add the loop to the experiment
    thisNB_ins_byp = NB_ins_byp.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisNB_ins_byp.rgb)
    if thisNB_ins_byp != None:
        for paramName in thisNB_ins_byp:
            globals()[paramName] = thisNB_ins_byp[paramName]
    
    for thisNB_ins_byp in NB_ins_byp:
        currentLoop = NB_ins_byp
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisNB_ins_byp.rgb)
        if thisNB_ins_byp != None:
            for paramName in thisNB_ins_byp:
                globals()[paramName] = thisNB_ins_byp[paramName]
        
        # --- Prepare to start Routine "NB_ins" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('NB_ins.started', globalClock.getTime(format='float'))
        # Run 'Begin Routine' code from nb_ins_code
        nb_ins_txt.alignText = 'center'
        # create starting attributes for nb_ins_key
        nb_ins_key.keys = []
        nb_ins_key.rt = []
        _nb_ins_key_allKeys = []
        # setup some python lists for storing info about the nb_ins_mse
        nb_ins_mse.x = []
        nb_ins_mse.y = []
        nb_ins_mse.leftButton = []
        nb_ins_mse.midButton = []
        nb_ins_mse.rightButton = []
        nb_ins_mse.time = []
        nb_ins_mse.clicked_name = []
        gotValidClick = False  # until a click is received
        # keep track of which components have finished
        NB_insComponents = [nb_ins_header, nb_ins_txt, nb_ins_key, nb_ins_btn, nb_ins_hint, nb_ins_mse]
        for thisComponent in NB_insComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "NB_ins" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *nb_ins_header* updates
            
            # if nb_ins_header is starting this frame...
            if nb_ins_header.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                nb_ins_header.frameNStart = frameN  # exact frame index
                nb_ins_header.tStart = t  # local t and not account for scr refresh
                nb_ins_header.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(nb_ins_header, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'nb_ins_header.started')
                # update status
                nb_ins_header.status = STARTED
                nb_ins_header.setAutoDraw(True)
            
            # if nb_ins_header is active this frame...
            if nb_ins_header.status == STARTED:
                # update params
                pass
            
            # *nb_ins_txt* updates
            
            # if nb_ins_txt is starting this frame...
            if nb_ins_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                nb_ins_txt.frameNStart = frameN  # exact frame index
                nb_ins_txt.tStart = t  # local t and not account for scr refresh
                nb_ins_txt.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(nb_ins_txt, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'nb_ins_txt.started')
                # update status
                nb_ins_txt.status = STARTED
                nb_ins_txt.setAutoDraw(True)
            
            # if nb_ins_txt is active this frame...
            if nb_ins_txt.status == STARTED:
                # update params
                pass
            
            # *nb_ins_key* updates
            waitOnFlip = False
            
            # if nb_ins_key is starting this frame...
            if nb_ins_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                nb_ins_key.frameNStart = frameN  # exact frame index
                nb_ins_key.tStart = t  # local t and not account for scr refresh
                nb_ins_key.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(nb_ins_key, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'nb_ins_key.started')
                # update status
                nb_ins_key.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(nb_ins_key.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(nb_ins_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if nb_ins_key.status == STARTED and not waitOnFlip:
                theseKeys = nb_ins_key.getKeys(keyList=['return','space'], ignoreKeys=["escape"], waitRelease=False)
                _nb_ins_key_allKeys.extend(theseKeys)
                if len(_nb_ins_key_allKeys):
                    nb_ins_key.keys = _nb_ins_key_allKeys[-1].name  # just the last key pressed
                    nb_ins_key.rt = _nb_ins_key_allKeys[-1].rt
                    nb_ins_key.duration = _nb_ins_key_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *nb_ins_btn* updates
            
            # if nb_ins_btn is starting this frame...
            if nb_ins_btn.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                nb_ins_btn.frameNStart = frameN  # exact frame index
                nb_ins_btn.tStart = t  # local t and not account for scr refresh
                nb_ins_btn.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(nb_ins_btn, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'nb_ins_btn.started')
                # update status
                nb_ins_btn.status = STARTED
                nb_ins_btn.setAutoDraw(True)
            
            # if nb_ins_btn is active this frame...
            if nb_ins_btn.status == STARTED:
                # update params
                pass
            
            # *nb_ins_hint* updates
            
            # if nb_ins_hint is starting this frame...
            if nb_ins_hint.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                nb_ins_hint.frameNStart = frameN  # exact frame index
                nb_ins_hint.tStart = t  # local t and not account for scr refresh
                nb_ins_hint.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(nb_ins_hint, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'nb_ins_hint.started')
                # update status
                nb_ins_hint.status = STARTED
                nb_ins_hint.setAutoDraw(True)
            
            # if nb_ins_hint is active this frame...
            if nb_ins_hint.status == STARTED:
                # update params
                pass
            # *nb_ins_mse* updates
            
            # if nb_ins_mse is starting this frame...
            if nb_ins_mse.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                nb_ins_mse.frameNStart = frameN  # exact frame index
                nb_ins_mse.tStart = t  # local t and not account for scr refresh
                nb_ins_mse.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(nb_ins_mse, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('nb_ins_mse.started', t)
                # update status
                nb_ins_mse.status = STARTED
                nb_ins_mse.mouseClock.reset()
                prevButtonState = nb_ins_mse.getPressed()  # if button is down already this ISN'T a new click
            if nb_ins_mse.status == STARTED:  # only update if started and not finished!
                buttons = nb_ins_mse.getPressed()
                if buttons != prevButtonState:  # button state changed?
                    prevButtonState = buttons
                    if sum(buttons) > 0:  # state changed to a new click
                        # check if the mouse was inside our 'clickable' objects
                        gotValidClick = False
                        clickableList = environmenttools.getFromNames(nb_ins_btn, namespace=locals())
                        for obj in clickableList:
                            # is this object clicked on?
                            if obj.contains(nb_ins_mse):
                                gotValidClick = True
                                nb_ins_mse.clicked_name.append(obj.name)
                        x, y = nb_ins_mse.getPos()
                        nb_ins_mse.x.append(x)
                        nb_ins_mse.y.append(y)
                        buttons = nb_ins_mse.getPressed()
                        nb_ins_mse.leftButton.append(buttons[0])
                        nb_ins_mse.midButton.append(buttons[1])
                        nb_ins_mse.rightButton.append(buttons[2])
                        nb_ins_mse.time.append(nb_ins_mse.mouseClock.getTime())
                        if gotValidClick:
                            continueRoutine = False  # end routine on response
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in NB_insComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "NB_ins" ---
        for thisComponent in NB_insComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('NB_ins.stopped', globalClock.getTime(format='float'))
        # check responses
        if nb_ins_key.keys in ['', [], None]:  # No response was made
            nb_ins_key.keys = None
        NB_ins_byp.addData('nb_ins_key.keys',nb_ins_key.keys)
        if nb_ins_key.keys != None:  # we had a response
            NB_ins_byp.addData('nb_ins_key.rt', nb_ins_key.rt)
            NB_ins_byp.addData('nb_ins_key.duration', nb_ins_key.duration)
        # store data for NB_ins_byp (TrialHandler)
        NB_ins_byp.addData('nb_ins_mse.x', nb_ins_mse.x)
        NB_ins_byp.addData('nb_ins_mse.y', nb_ins_mse.y)
        NB_ins_byp.addData('nb_ins_mse.leftButton', nb_ins_mse.leftButton)
        NB_ins_byp.addData('nb_ins_mse.midButton', nb_ins_mse.midButton)
        NB_ins_byp.addData('nb_ins_mse.rightButton', nb_ins_mse.rightButton)
        NB_ins_byp.addData('nb_ins_mse.time', nb_ins_mse.time)
        NB_ins_byp.addData('nb_ins_mse.clicked_name', nb_ins_mse.clicked_name)
        # the Routine "NB_ins" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
    # completed 1.0 repeats of 'NB_ins_byp'
    
    
    # set up handler to look after randomisation of conditions etc
    NB_byp = data.TrialHandler(nReps=1.0, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='NB_byp')
    thisExp.addLoop(NB_byp)  # add the loop to the experiment
    thisNB_byp = NB_byp.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisNB_byp.rgb)
    if thisNB_byp != None:
        for paramName in thisNB_byp:
            globals()[paramName] = thisNB_byp[paramName]
    
    for thisNB_byp in NB_byp:
        currentLoop = NB_byp
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisNB_byp.rgb)
        if thisNB_byp != None:
            for paramName in thisNB_byp:
                globals()[paramName] = thisNB_byp[paramName]
        
        # set up handler to look after randomisation of conditions etc
        NB_loop = data.TrialHandler(nReps=NB_trials, method='random', 
            extraInfo=expInfo, originPath=-1,
            trialList=[None],
            seed=None, name='NB_loop')
        thisExp.addLoop(NB_loop)  # add the loop to the experiment
        thisNB_loop = NB_loop.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisNB_loop.rgb)
        if thisNB_loop != None:
            for paramName in thisNB_loop:
                globals()[paramName] = thisNB_loop[paramName]
        
        for thisNB_loop in NB_loop:
            currentLoop = NB_loop
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
            )
            # abbreviate parameter names if possible (e.g. rgb = thisNB_loop.rgb)
            if thisNB_loop != None:
                for paramName in thisNB_loop:
                    globals()[paramName] = thisNB_loop[paramName]
            
            # --- Prepare to start Routine "go_NB3min" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('go_NB3min.started', globalClock.getTime(format='float'))
            # keep track of which components have finished
            go_NB3minComponents = [startNB3min_txt]
            for thisComponent in go_NB3minComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "go_NB3min" ---
            routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 1.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *startNB3min_txt* updates
                
                # if startNB3min_txt is starting this frame...
                if startNB3min_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    startNB3min_txt.frameNStart = frameN  # exact frame index
                    startNB3min_txt.tStart = t  # local t and not account for scr refresh
                    startNB3min_txt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(startNB3min_txt, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'startNB3min_txt.started')
                    # update status
                    startNB3min_txt.status = STARTED
                    startNB3min_txt.setAutoDraw(True)
                
                # if startNB3min_txt is active this frame...
                if startNB3min_txt.status == STARTED:
                    # update params
                    pass
                
                # if startNB3min_txt is stopping this frame...
                if startNB3min_txt.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > startNB3min_txt.tStartRefresh + 1-frameTolerance:
                        # keep track of stop time/frame for later
                        startNB3min_txt.tStop = t  # not accounting for scr refresh
                        startNB3min_txt.tStopRefresh = tThisFlipGlobal  # on global time
                        startNB3min_txt.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'startNB3min_txt.stopped')
                        # update status
                        startNB3min_txt.status = FINISHED
                        startNB3min_txt.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in go_NB3minComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "go_NB3min" ---
            for thisComponent in go_NB3minComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('go_NB3min.stopped', globalClock.getTime(format='float'))
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if routineForceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-1.000000)
            
            # --- Prepare to start Routine "NB_3min" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('NB_3min.started', globalClock.getTime(format='float'))
            # Run 'Begin Routine' code from NB3min_code
            ##  marker: NB3min START  ##
            Marker.setData(21); core.wait(0.1); Marker.setData(0)
            # keep track of which components have finished
            NB_3minComponents = [NB3min_txt, NB3min_countdown]
            for thisComponent in NB_3minComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "NB_3min" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *NB3min_txt* updates
                
                # if NB3min_txt is starting this frame...
                if NB3min_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    NB3min_txt.frameNStart = frameN  # exact frame index
                    NB3min_txt.tStart = t  # local t and not account for scr refresh
                    NB3min_txt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(NB3min_txt, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'NB3min_txt.started')
                    # update status
                    NB3min_txt.status = STARTED
                    NB3min_txt.setAutoDraw(True)
                
                # if NB3min_txt is active this frame...
                if NB3min_txt.status == STARTED:
                    # update params
                    pass
                
                # if NB3min_txt is stopping this frame...
                if NB3min_txt.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > NB3min_txt.tStartRefresh + breathing_dur-frameTolerance:
                        # keep track of stop time/frame for later
                        NB3min_txt.tStop = t  # not accounting for scr refresh
                        NB3min_txt.tStopRefresh = tThisFlipGlobal  # on global time
                        NB3min_txt.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'NB3min_txt.stopped')
                        # update status
                        NB3min_txt.status = FINISHED
                        NB3min_txt.setAutoDraw(False)
                
                # *NB3min_countdown* updates
                
                # if NB3min_countdown is starting this frame...
                if NB3min_countdown.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    NB3min_countdown.frameNStart = frameN  # exact frame index
                    NB3min_countdown.tStart = t  # local t and not account for scr refresh
                    NB3min_countdown.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(NB3min_countdown, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'NB3min_countdown.started')
                    # update status
                    NB3min_countdown.status = STARTED
                    NB3min_countdown.setAutoDraw(True)
                
                # if NB3min_countdown is active this frame...
                if NB3min_countdown.status == STARTED:
                    # update params
                    NB3min_countdown.setText(str(breathing_dur-int(t)), log=False)
                
                # if NB3min_countdown is stopping this frame...
                if NB3min_countdown.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > NB3min_countdown.tStartRefresh + breathing_dur-frameTolerance:
                        # keep track of stop time/frame for later
                        NB3min_countdown.tStop = t  # not accounting for scr refresh
                        NB3min_countdown.tStopRefresh = tThisFlipGlobal  # on global time
                        NB3min_countdown.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'NB3min_countdown.stopped')
                        # update status
                        NB3min_countdown.status = FINISHED
                        NB3min_countdown.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in NB_3minComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "NB_3min" ---
            for thisComponent in NB_3minComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('NB_3min.stopped', globalClock.getTime(format='float'))
            # Run 'End Routine' code from NB3min_code
            ##  marker: NB3min STOP  ##
            Marker.setData(23); core.wait(0.1); Marker.setData(0)
            # the Routine "NB_3min" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
        # completed NB_trials repeats of 'NB_loop'
        
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'NB_byp'
    
    
    # set up handler to look after randomisation of conditions etc
    crea_tasks_byp = data.TrialHandler(nReps=1.0, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='crea_tasks_byp')
    thisExp.addLoop(crea_tasks_byp)  # add the loop to the experiment
    thisCrea_tasks_byp = crea_tasks_byp.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisCrea_tasks_byp.rgb)
    if thisCrea_tasks_byp != None:
        for paramName in thisCrea_tasks_byp:
            globals()[paramName] = thisCrea_tasks_byp[paramName]
    
    for thisCrea_tasks_byp in crea_tasks_byp:
        currentLoop = crea_tasks_byp
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisCrea_tasks_byp.rgb)
        if thisCrea_tasks_byp != None:
            for paramName in thisCrea_tasks_byp:
                globals()[paramName] = thisCrea_tasks_byp[paramName]
        
        # --- Prepare to start Routine "descCreaTasks" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('descCreaTasks.started', globalClock.getTime(format='float'))
        # Run 'Begin Routine' code from crea_tasks_code
        crea_tasks_txt.alignText = 'left'
        # create starting attributes for crea_tasks_key
        crea_tasks_key.keys = []
        crea_tasks_key.rt = []
        _crea_tasks_key_allKeys = []
        # setup some python lists for storing info about the crea_tasks_mse
        crea_tasks_mse.x = []
        crea_tasks_mse.y = []
        crea_tasks_mse.leftButton = []
        crea_tasks_mse.midButton = []
        crea_tasks_mse.rightButton = []
        crea_tasks_mse.time = []
        crea_tasks_mse.clicked_name = []
        gotValidClick = False  # until a click is received
        # keep track of which components have finished
        descCreaTasksComponents = [crea_tasks_header, crea_tasks_txt, crea_tasks_key, crea_tasks_btn, crea_tasks_hint, crea_tasks_mse]
        for thisComponent in descCreaTasksComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "descCreaTasks" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *crea_tasks_header* updates
            
            # if crea_tasks_header is starting this frame...
            if crea_tasks_header.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                crea_tasks_header.frameNStart = frameN  # exact frame index
                crea_tasks_header.tStart = t  # local t and not account for scr refresh
                crea_tasks_header.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(crea_tasks_header, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'crea_tasks_header.started')
                # update status
                crea_tasks_header.status = STARTED
                crea_tasks_header.setAutoDraw(True)
            
            # if crea_tasks_header is active this frame...
            if crea_tasks_header.status == STARTED:
                # update params
                pass
            
            # if crea_tasks_header is stopping this frame...
            if crea_tasks_header.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > crea_tasks_header.tStartRefresh + descCrea_dur-frameTolerance:
                    # keep track of stop time/frame for later
                    crea_tasks_header.tStop = t  # not accounting for scr refresh
                    crea_tasks_header.tStopRefresh = tThisFlipGlobal  # on global time
                    crea_tasks_header.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'crea_tasks_header.stopped')
                    # update status
                    crea_tasks_header.status = FINISHED
                    crea_tasks_header.setAutoDraw(False)
            
            # *crea_tasks_txt* updates
            
            # if crea_tasks_txt is starting this frame...
            if crea_tasks_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                crea_tasks_txt.frameNStart = frameN  # exact frame index
                crea_tasks_txt.tStart = t  # local t and not account for scr refresh
                crea_tasks_txt.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(crea_tasks_txt, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'crea_tasks_txt.started')
                # update status
                crea_tasks_txt.status = STARTED
                crea_tasks_txt.setAutoDraw(True)
            
            # if crea_tasks_txt is active this frame...
            if crea_tasks_txt.status == STARTED:
                # update params
                pass
            
            # if crea_tasks_txt is stopping this frame...
            if crea_tasks_txt.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > crea_tasks_txt.tStartRefresh + descCrea_dur-frameTolerance:
                    # keep track of stop time/frame for later
                    crea_tasks_txt.tStop = t  # not accounting for scr refresh
                    crea_tasks_txt.tStopRefresh = tThisFlipGlobal  # on global time
                    crea_tasks_txt.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'crea_tasks_txt.stopped')
                    # update status
                    crea_tasks_txt.status = FINISHED
                    crea_tasks_txt.setAutoDraw(False)
            
            # *crea_tasks_key* updates
            waitOnFlip = False
            
            # if crea_tasks_key is starting this frame...
            if crea_tasks_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                crea_tasks_key.frameNStart = frameN  # exact frame index
                crea_tasks_key.tStart = t  # local t and not account for scr refresh
                crea_tasks_key.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(crea_tasks_key, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'crea_tasks_key.started')
                # update status
                crea_tasks_key.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(crea_tasks_key.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(crea_tasks_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if crea_tasks_key is stopping this frame...
            if crea_tasks_key.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > crea_tasks_key.tStartRefresh + descCrea_dur-frameTolerance:
                    # keep track of stop time/frame for later
                    crea_tasks_key.tStop = t  # not accounting for scr refresh
                    crea_tasks_key.tStopRefresh = tThisFlipGlobal  # on global time
                    crea_tasks_key.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'crea_tasks_key.stopped')
                    # update status
                    crea_tasks_key.status = FINISHED
                    crea_tasks_key.status = FINISHED
            if crea_tasks_key.status == STARTED and not waitOnFlip:
                theseKeys = crea_tasks_key.getKeys(keyList=['return','space'], ignoreKeys=["escape"], waitRelease=False)
                _crea_tasks_key_allKeys.extend(theseKeys)
                if len(_crea_tasks_key_allKeys):
                    crea_tasks_key.keys = _crea_tasks_key_allKeys[-1].name  # just the last key pressed
                    crea_tasks_key.rt = _crea_tasks_key_allKeys[-1].rt
                    crea_tasks_key.duration = _crea_tasks_key_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *crea_tasks_btn* updates
            
            # if crea_tasks_btn is starting this frame...
            if crea_tasks_btn.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                crea_tasks_btn.frameNStart = frameN  # exact frame index
                crea_tasks_btn.tStart = t  # local t and not account for scr refresh
                crea_tasks_btn.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(crea_tasks_btn, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'crea_tasks_btn.started')
                # update status
                crea_tasks_btn.status = STARTED
                crea_tasks_btn.setAutoDraw(True)
            
            # if crea_tasks_btn is active this frame...
            if crea_tasks_btn.status == STARTED:
                # update params
                pass
            
            # if crea_tasks_btn is stopping this frame...
            if crea_tasks_btn.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > crea_tasks_btn.tStartRefresh + descCrea_dur-frameTolerance:
                    # keep track of stop time/frame for later
                    crea_tasks_btn.tStop = t  # not accounting for scr refresh
                    crea_tasks_btn.tStopRefresh = tThisFlipGlobal  # on global time
                    crea_tasks_btn.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'crea_tasks_btn.stopped')
                    # update status
                    crea_tasks_btn.status = FINISHED
                    crea_tasks_btn.setAutoDraw(False)
            
            # *crea_tasks_hint* updates
            
            # if crea_tasks_hint is starting this frame...
            if crea_tasks_hint.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                crea_tasks_hint.frameNStart = frameN  # exact frame index
                crea_tasks_hint.tStart = t  # local t and not account for scr refresh
                crea_tasks_hint.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(crea_tasks_hint, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'crea_tasks_hint.started')
                # update status
                crea_tasks_hint.status = STARTED
                crea_tasks_hint.setAutoDraw(True)
            
            # if crea_tasks_hint is active this frame...
            if crea_tasks_hint.status == STARTED:
                # update params
                pass
            
            # if crea_tasks_hint is stopping this frame...
            if crea_tasks_hint.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > crea_tasks_hint.tStartRefresh + descCrea_dur-frameTolerance:
                    # keep track of stop time/frame for later
                    crea_tasks_hint.tStop = t  # not accounting for scr refresh
                    crea_tasks_hint.tStopRefresh = tThisFlipGlobal  # on global time
                    crea_tasks_hint.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'crea_tasks_hint.stopped')
                    # update status
                    crea_tasks_hint.status = FINISHED
                    crea_tasks_hint.setAutoDraw(False)
            # *crea_tasks_mse* updates
            
            # if crea_tasks_mse is starting this frame...
            if crea_tasks_mse.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                crea_tasks_mse.frameNStart = frameN  # exact frame index
                crea_tasks_mse.tStart = t  # local t and not account for scr refresh
                crea_tasks_mse.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(crea_tasks_mse, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('crea_tasks_mse.started', t)
                # update status
                crea_tasks_mse.status = STARTED
                crea_tasks_mse.mouseClock.reset()
                prevButtonState = crea_tasks_mse.getPressed()  # if button is down already this ISN'T a new click
            
            # if crea_tasks_mse is stopping this frame...
            if crea_tasks_mse.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > crea_tasks_mse.tStartRefresh + descCrea_dur-frameTolerance:
                    # keep track of stop time/frame for later
                    crea_tasks_mse.tStop = t  # not accounting for scr refresh
                    crea_tasks_mse.tStopRefresh = tThisFlipGlobal  # on global time
                    crea_tasks_mse.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.addData('crea_tasks_mse.stopped', t)
                    # update status
                    crea_tasks_mse.status = FINISHED
            if crea_tasks_mse.status == STARTED:  # only update if started and not finished!
                buttons = crea_tasks_mse.getPressed()
                if buttons != prevButtonState:  # button state changed?
                    prevButtonState = buttons
                    if sum(buttons) > 0:  # state changed to a new click
                        # check if the mouse was inside our 'clickable' objects
                        gotValidClick = False
                        clickableList = environmenttools.getFromNames(crea_tasks_btn, namespace=locals())
                        for obj in clickableList:
                            # is this object clicked on?
                            if obj.contains(crea_tasks_mse):
                                gotValidClick = True
                                crea_tasks_mse.clicked_name.append(obj.name)
                        x, y = crea_tasks_mse.getPos()
                        crea_tasks_mse.x.append(x)
                        crea_tasks_mse.y.append(y)
                        buttons = crea_tasks_mse.getPressed()
                        crea_tasks_mse.leftButton.append(buttons[0])
                        crea_tasks_mse.midButton.append(buttons[1])
                        crea_tasks_mse.rightButton.append(buttons[2])
                        crea_tasks_mse.time.append(crea_tasks_mse.mouseClock.getTime())
                        if gotValidClick:
                            continueRoutine = False  # end routine on response
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in descCreaTasksComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "descCreaTasks" ---
        for thisComponent in descCreaTasksComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('descCreaTasks.stopped', globalClock.getTime(format='float'))
        # check responses
        if crea_tasks_key.keys in ['', [], None]:  # No response was made
            crea_tasks_key.keys = None
        crea_tasks_byp.addData('crea_tasks_key.keys',crea_tasks_key.keys)
        if crea_tasks_key.keys != None:  # we had a response
            crea_tasks_byp.addData('crea_tasks_key.rt', crea_tasks_key.rt)
            crea_tasks_byp.addData('crea_tasks_key.duration', crea_tasks_key.duration)
        # store data for crea_tasks_byp (TrialHandler)
        crea_tasks_byp.addData('crea_tasks_mse.x', crea_tasks_mse.x)
        crea_tasks_byp.addData('crea_tasks_mse.y', crea_tasks_mse.y)
        crea_tasks_byp.addData('crea_tasks_mse.leftButton', crea_tasks_mse.leftButton)
        crea_tasks_byp.addData('crea_tasks_mse.midButton', crea_tasks_mse.midButton)
        crea_tasks_byp.addData('crea_tasks_mse.rightButton', crea_tasks_mse.rightButton)
        crea_tasks_byp.addData('crea_tasks_mse.time', crea_tasks_mse.time)
        crea_tasks_byp.addData('crea_tasks_mse.clicked_name', crea_tasks_mse.clicked_name)
        # the Routine "descCreaTasks" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
    # completed 1.0 repeats of 'crea_tasks_byp'
    
    
    # set up handler to look after randomisation of conditions etc
    crea_blocks = data.TrialHandler(nReps=1.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('stim/crea_blocks.csv'),
        seed=None, name='crea_blocks')
    thisExp.addLoop(crea_blocks)  # add the loop to the experiment
    thisCrea_block = crea_blocks.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisCrea_block.rgb)
    if thisCrea_block != None:
        for paramName in thisCrea_block:
            globals()[paramName] = thisCrea_block[paramName]
    
    for thisCrea_block in crea_blocks:
        currentLoop = crea_blocks
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisCrea_block.rgb)
        if thisCrea_block != None:
            for paramName in thisCrea_block:
                globals()[paramName] = thisCrea_block[paramName]
        
        # --- Prepare to start Routine "crea_items_count" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('crea_items_count.started', globalClock.getTime(format='float'))
        # Run 'Begin Routine' code from cic_code
        if crea_blocks.thisN < 7:
            go_NB1min = 1
        else:
            go_NB1min = 0
        # keep track of which components have finished
        crea_items_countComponents = []
        for thisComponent in crea_items_countComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "crea_items_count" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in crea_items_countComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "crea_items_count" ---
        for thisComponent in crea_items_countComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('crea_items_count.stopped', globalClock.getTime(format='float'))
        # the Routine "crea_items_count" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        byp_AUTinfo = data.TrialHandler(nReps=0.0, method='sequential', 
            extraInfo=expInfo, originPath=-1,
            trialList=[None],
            seed=None, name='byp_AUTinfo')
        thisExp.addLoop(byp_AUTinfo)  # add the loop to the experiment
        thisByp_AUTinfo = byp_AUTinfo.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisByp_AUTinfo.rgb)
        if thisByp_AUTinfo != None:
            for paramName in thisByp_AUTinfo:
                globals()[paramName] = thisByp_AUTinfo[paramName]
        
        for thisByp_AUTinfo in byp_AUTinfo:
            currentLoop = byp_AUTinfo
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
            )
            # abbreviate parameter names if possible (e.g. rgb = thisByp_AUTinfo.rgb)
            if thisByp_AUTinfo != None:
                for paramName in thisByp_AUTinfo:
                    globals()[paramName] = thisByp_AUTinfo[paramName]
            
            # --- Prepare to start Routine "AUT_info" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('AUT_info.started', globalClock.getTime(format='float'))
            # Run 'Begin Routine' code from AUT_info_code
            print(" ")
            print(f'{crea_blocks=}')
            print(f'{crea_blocks.trialList=}')
            print(f'{crea_blocks.trialList[0]=}')
            print(f'{AUT_run=} | {TTCT_run=}'); 
            print(" ")
            # keep track of which components have finished
            AUT_infoComponents = []
            for thisComponent in AUT_infoComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "AUT_info" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in AUT_infoComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "AUT_info" ---
            for thisComponent in AUT_infoComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('AUT_info.stopped', globalClock.getTime(format='float'))
            # the Routine "AUT_info" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
        # completed 0.0 repeats of 'byp_AUTinfo'
        
        
        # set up handler to look after randomisation of conditions etc
        AUT_trials = data.TrialHandler(nReps=AUT_run, method='random', 
            extraInfo=expInfo, originPath=-1,
            trialList=[None],
            seed=None, name='AUT_trials')
        thisExp.addLoop(AUT_trials)  # add the loop to the experiment
        thisAUT_trial = AUT_trials.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisAUT_trial.rgb)
        if thisAUT_trial != None:
            for paramName in thisAUT_trial:
                globals()[paramName] = thisAUT_trial[paramName]
        
        for thisAUT_trial in AUT_trials:
            currentLoop = AUT_trials
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
            )
            # abbreviate parameter names if possible (e.g. rgb = thisAUT_trial.rgb)
            if thisAUT_trial != None:
                for paramName in thisAUT_trial:
                    globals()[paramName] = thisAUT_trial[paramName]
            
            # set up handler to look after randomisation of conditions etc
            AUTitem_loop = data.TrialHandler(nReps=1, method='random', 
                extraInfo=expInfo, originPath=-1,
                trialList=[None],
                seed=None, name='AUTitem_loop')
            thisExp.addLoop(AUTitem_loop)  # add the loop to the experiment
            thisAUTitem_loop = AUTitem_loop.trialList[0]  # so we can initialise stimuli with some values
            # abbreviate parameter names if possible (e.g. rgb = thisAUTitem_loop.rgb)
            if thisAUTitem_loop != None:
                for paramName in thisAUTitem_loop:
                    globals()[paramName] = thisAUTitem_loop[paramName]
            
            for thisAUTitem_loop in AUTitem_loop:
                currentLoop = AUTitem_loop
                thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                )
                # abbreviate parameter names if possible (e.g. rgb = thisAUTitem_loop.rgb)
                if thisAUTitem_loop != None:
                    for paramName in thisAUTitem_loop:
                        globals()[paramName] = thisAUTitem_loop[paramName]
                
                # --- Prepare to start Routine "AUTtrial" ---
                continueRoutine = True
                # update component parameters for each repeat
                thisExp.addData('AUTtrial.started', globalClock.getTime(format='float'))
                # Run 'Begin Routine' code from AUT_code
                ##  marker: AUTtrials START  ##
                Marker.setData(51); core.wait(0.1); Marker.setData(0)
                
                AUTitem = AUTitems_block1[presented_AUTitems]
                #print(f'{AUTitem=}')
                AUT_trials.addData('AUTitem', AUTitem)  #’trials’=current loop
                
                AUT_item.setText(AUTitem)
                # create starting attributes for AUT_idea_key
                AUT_idea_key.keys = []
                AUT_idea_key.rt = []
                _AUT_idea_key_allKeys = []
                # setup some python lists for storing info about the AUT_idea_mse
                AUT_idea_mse.x = []
                AUT_idea_mse.y = []
                AUT_idea_mse.leftButton = []
                AUT_idea_mse.midButton = []
                AUT_idea_mse.rightButton = []
                AUT_idea_mse.time = []
                AUT_idea_mse.clicked_name = []
                gotValidClick = False  # until a click is received
                # keep track of which components have finished
                AUTtrialComponents = [AUT_fix, AUT_mrk, AUT_item, AUT_idea_key, AUT_idea_btn, AUT_idea_txt, AUT_idea_mse]
                for thisComponent in AUTtrialComponents:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "AUTtrial" ---
                routineForceEnded = not continueRoutine
                while continueRoutine:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *AUT_fix* updates
                    
                    # if AUT_fix is starting this frame...
                    if AUT_fix.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        AUT_fix.frameNStart = frameN  # exact frame index
                        AUT_fix.tStart = t  # local t and not account for scr refresh
                        AUT_fix.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(AUT_fix, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'AUT_fix.started')
                        # update status
                        AUT_fix.status = STARTED
                        AUT_fix.setAutoDraw(True)
                    
                    # if AUT_fix is active this frame...
                    if AUT_fix.status == STARTED:
                        # update params
                        pass
                    
                    # if AUT_fix is stopping this frame...
                    if AUT_fix.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > AUT_fix.tStartRefresh + fixDur-frameTolerance:
                            # keep track of stop time/frame for later
                            AUT_fix.tStop = t  # not accounting for scr refresh
                            AUT_fix.tStopRefresh = tThisFlipGlobal  # on global time
                            AUT_fix.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'AUT_fix.stopped')
                            # update status
                            AUT_fix.status = FINISHED
                            AUT_fix.setAutoDraw(False)
                    # *AUT_mrk* updates
                    
                    # if AUT_mrk is starting this frame...
                    if AUT_mrk.status == NOT_STARTED and AUT_fix.status==FINISHED:
                        # keep track of start time/frame for later
                        AUT_mrk.frameNStart = frameN  # exact frame index
                        AUT_mrk.tStart = t  # local t and not account for scr refresh
                        AUT_mrk.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(AUT_mrk, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.addData('AUT_mrk.started', t)
                        # update status
                        AUT_mrk.status = STARTED
                        AUT_mrk.status = STARTED
                        win.callOnFlip(AUT_mrk.setData, int(23))
                    
                    # if AUT_mrk is stopping this frame...
                    if AUT_mrk.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > AUT_mrk.tStartRefresh + 0.1-frameTolerance:
                            # keep track of stop time/frame for later
                            AUT_mrk.tStop = t  # not accounting for scr refresh
                            AUT_mrk.tStopRefresh = tThisFlipGlobal  # on global time
                            AUT_mrk.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.addData('AUT_mrk.stopped', t)
                            # update status
                            AUT_mrk.status = FINISHED
                            win.callOnFlip(AUT_mrk.setData, int(0))
                    
                    # *AUT_item* updates
                    
                    # if AUT_item is starting this frame...
                    if AUT_item.status == NOT_STARTED and AUT_fix.status==FINISHED:
                        # keep track of start time/frame for later
                        AUT_item.frameNStart = frameN  # exact frame index
                        AUT_item.tStart = t  # local t and not account for scr refresh
                        AUT_item.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(AUT_item, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'AUT_item.started')
                        # update status
                        AUT_item.status = STARTED
                        AUT_item.setAutoDraw(True)
                    
                    # if AUT_item is active this frame...
                    if AUT_item.status == STARTED:
                        # update params
                        pass
                    
                    # if AUT_item is stopping this frame...
                    if AUT_item.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > AUT_item.tStartRefresh + AUTitem_dur-frameTolerance:
                            # keep track of stop time/frame for later
                            AUT_item.tStop = t  # not accounting for scr refresh
                            AUT_item.tStopRefresh = tThisFlipGlobal  # on global time
                            AUT_item.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'AUT_item.stopped')
                            # update status
                            AUT_item.status = FINISHED
                            AUT_item.setAutoDraw(False)
                    
                    # *AUT_idea_key* updates
                    waitOnFlip = False
                    
                    # if AUT_idea_key is starting this frame...
                    if AUT_idea_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        AUT_idea_key.frameNStart = frameN  # exact frame index
                        AUT_idea_key.tStart = t  # local t and not account for scr refresh
                        AUT_idea_key.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(AUT_idea_key, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'AUT_idea_key.started')
                        # update status
                        AUT_idea_key.status = STARTED
                        # keyboard checking is just starting
                        waitOnFlip = True
                        win.callOnFlip(AUT_idea_key.clock.reset)  # t=0 on next screen flip
                        win.callOnFlip(AUT_idea_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
                    
                    # if AUT_idea_key is stopping this frame...
                    if AUT_idea_key.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > AUT_idea_key.tStartRefresh + AUTitem_dur-frameTolerance:
                            # keep track of stop time/frame for later
                            AUT_idea_key.tStop = t  # not accounting for scr refresh
                            AUT_idea_key.tStopRefresh = tThisFlipGlobal  # on global time
                            AUT_idea_key.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'AUT_idea_key.stopped')
                            # update status
                            AUT_idea_key.status = FINISHED
                            AUT_idea_key.status = FINISHED
                    if AUT_idea_key.status == STARTED and not waitOnFlip:
                        theseKeys = AUT_idea_key.getKeys(keyList=['return','space'], ignoreKeys=["escape"], waitRelease=False)
                        _AUT_idea_key_allKeys.extend(theseKeys)
                        if len(_AUT_idea_key_allKeys):
                            AUT_idea_key.keys = _AUT_idea_key_allKeys[-1].name  # just the last key pressed
                            AUT_idea_key.rt = _AUT_idea_key_allKeys[-1].rt
                            AUT_idea_key.duration = _AUT_idea_key_allKeys[-1].duration
                            # a response ends the routine
                            continueRoutine = False
                    
                    # *AUT_idea_btn* updates
                    
                    # if AUT_idea_btn is starting this frame...
                    if AUT_idea_btn.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        AUT_idea_btn.frameNStart = frameN  # exact frame index
                        AUT_idea_btn.tStart = t  # local t and not account for scr refresh
                        AUT_idea_btn.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(AUT_idea_btn, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'AUT_idea_btn.started')
                        # update status
                        AUT_idea_btn.status = STARTED
                        AUT_idea_btn.setAutoDraw(True)
                    
                    # if AUT_idea_btn is active this frame...
                    if AUT_idea_btn.status == STARTED:
                        # update params
                        pass
                    
                    # if AUT_idea_btn is stopping this frame...
                    if AUT_idea_btn.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > AUT_idea_btn.tStartRefresh + AUTitem_dur-frameTolerance:
                            # keep track of stop time/frame for later
                            AUT_idea_btn.tStop = t  # not accounting for scr refresh
                            AUT_idea_btn.tStopRefresh = tThisFlipGlobal  # on global time
                            AUT_idea_btn.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'AUT_idea_btn.stopped')
                            # update status
                            AUT_idea_btn.status = FINISHED
                            AUT_idea_btn.setAutoDraw(False)
                    
                    # *AUT_idea_txt* updates
                    
                    # if AUT_idea_txt is starting this frame...
                    if AUT_idea_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        AUT_idea_txt.frameNStart = frameN  # exact frame index
                        AUT_idea_txt.tStart = t  # local t and not account for scr refresh
                        AUT_idea_txt.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(AUT_idea_txt, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'AUT_idea_txt.started')
                        # update status
                        AUT_idea_txt.status = STARTED
                        AUT_idea_txt.setAutoDraw(True)
                    
                    # if AUT_idea_txt is active this frame...
                    if AUT_idea_txt.status == STARTED:
                        # update params
                        pass
                    
                    # if AUT_idea_txt is stopping this frame...
                    if AUT_idea_txt.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > AUT_idea_txt.tStartRefresh + AUTitem_dur-frameTolerance:
                            # keep track of stop time/frame for later
                            AUT_idea_txt.tStop = t  # not accounting for scr refresh
                            AUT_idea_txt.tStopRefresh = tThisFlipGlobal  # on global time
                            AUT_idea_txt.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'AUT_idea_txt.stopped')
                            # update status
                            AUT_idea_txt.status = FINISHED
                            AUT_idea_txt.setAutoDraw(False)
                    # *AUT_idea_mse* updates
                    
                    # if AUT_idea_mse is starting this frame...
                    if AUT_idea_mse.status == NOT_STARTED and t >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        AUT_idea_mse.frameNStart = frameN  # exact frame index
                        AUT_idea_mse.tStart = t  # local t and not account for scr refresh
                        AUT_idea_mse.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(AUT_idea_mse, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.addData('AUT_idea_mse.started', t)
                        # update status
                        AUT_idea_mse.status = STARTED
                        AUT_idea_mse.mouseClock.reset()
                        prevButtonState = AUT_idea_mse.getPressed()  # if button is down already this ISN'T a new click
                    
                    # if AUT_idea_mse is stopping this frame...
                    if AUT_idea_mse.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > AUT_idea_mse.tStartRefresh + AUTitem_dur-frameTolerance:
                            # keep track of stop time/frame for later
                            AUT_idea_mse.tStop = t  # not accounting for scr refresh
                            AUT_idea_mse.tStopRefresh = tThisFlipGlobal  # on global time
                            AUT_idea_mse.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.addData('AUT_idea_mse.stopped', t)
                            # update status
                            AUT_idea_mse.status = FINISHED
                    if AUT_idea_mse.status == STARTED:  # only update if started and not finished!
                        buttons = AUT_idea_mse.getPressed()
                        if buttons != prevButtonState:  # button state changed?
                            prevButtonState = buttons
                            if sum(buttons) > 0:  # state changed to a new click
                                # check if the mouse was inside our 'clickable' objects
                                gotValidClick = False
                                clickableList = environmenttools.getFromNames(AUT_idea_btn, namespace=locals())
                                for obj in clickableList:
                                    # is this object clicked on?
                                    if obj.contains(AUT_idea_mse):
                                        gotValidClick = True
                                        AUT_idea_mse.clicked_name.append(obj.name)
                                x, y = AUT_idea_mse.getPos()
                                AUT_idea_mse.x.append(x)
                                AUT_idea_mse.y.append(y)
                                buttons = AUT_idea_mse.getPressed()
                                AUT_idea_mse.leftButton.append(buttons[0])
                                AUT_idea_mse.midButton.append(buttons[1])
                                AUT_idea_mse.rightButton.append(buttons[2])
                                AUT_idea_mse.time.append(AUT_idea_mse.mouseClock.getTime())
                                if gotValidClick:
                                    continueRoutine = False  # end routine on response
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in AUTtrialComponents:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "AUTtrial" ---
                for thisComponent in AUTtrialComponents:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                thisExp.addData('AUTtrial.stopped', globalClock.getTime(format='float'))
                # Run 'End Routine' code from AUT_code
                presented_AUTitems += 1
                #print(f'{presented_AUTitems=}')
                
                if AUT_mrk.status == STARTED:
                    win.callOnFlip(AUT_mrk.setData, int(0))
                # check responses
                if AUT_idea_key.keys in ['', [], None]:  # No response was made
                    AUT_idea_key.keys = None
                AUTitem_loop.addData('AUT_idea_key.keys',AUT_idea_key.keys)
                if AUT_idea_key.keys != None:  # we had a response
                    AUTitem_loop.addData('AUT_idea_key.rt', AUT_idea_key.rt)
                    AUTitem_loop.addData('AUT_idea_key.duration', AUT_idea_key.duration)
                # store data for AUTitem_loop (TrialHandler)
                AUTitem_loop.addData('AUT_idea_mse.x', AUT_idea_mse.x)
                AUTitem_loop.addData('AUT_idea_mse.y', AUT_idea_mse.y)
                AUTitem_loop.addData('AUT_idea_mse.leftButton', AUT_idea_mse.leftButton)
                AUTitem_loop.addData('AUT_idea_mse.midButton', AUT_idea_mse.midButton)
                AUTitem_loop.addData('AUT_idea_mse.rightButton', AUT_idea_mse.rightButton)
                AUTitem_loop.addData('AUT_idea_mse.time', AUT_idea_mse.time)
                AUTitem_loop.addData('AUT_idea_mse.clicked_name', AUT_idea_mse.clicked_name)
                # the Routine "AUTtrial" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
                
                # --- Prepare to start Routine "t_AUTanswer" ---
                continueRoutine = True
                # update component parameters for each repeat
                thisExp.addData('t_AUTanswer.started', globalClock.getTime(format='float'))
                # Run 'Begin Routine' code from AUTanswer_code
                ##  marker: AUTanswer START  ##
                Marker.setData(53); core.wait(0.1); Marker.setData(0)
                answer_box.reset()
                answer_box.setText('')
                # setup some python lists for storing info about the go_on_mse
                go_on_mse.x = []
                go_on_mse.y = []
                go_on_mse.leftButton = []
                go_on_mse.midButton = []
                go_on_mse.rightButton = []
                go_on_mse.time = []
                go_on_mse.clicked_name = []
                gotValidClick = False  # until a click is received
                # keep track of which components have finished
                t_AUTanswerComponents = [ans_prompt, answer_box, go_on_btn, go_on_txt, go_on_mse]
                for thisComponent in t_AUTanswerComponents:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "t_AUTanswer" ---
                routineForceEnded = not continueRoutine
                while continueRoutine:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *ans_prompt* updates
                    
                    # if ans_prompt is starting this frame...
                    if ans_prompt.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                        # keep track of start time/frame for later
                        ans_prompt.frameNStart = frameN  # exact frame index
                        ans_prompt.tStart = t  # local t and not account for scr refresh
                        ans_prompt.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(ans_prompt, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'ans_prompt.started')
                        # update status
                        ans_prompt.status = STARTED
                        ans_prompt.setAutoDraw(True)
                    
                    # if ans_prompt is active this frame...
                    if ans_prompt.status == STARTED:
                        # update params
                        pass
                    
                    # *answer_box* updates
                    
                    # if answer_box is starting this frame...
                    if answer_box.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        answer_box.frameNStart = frameN  # exact frame index
                        answer_box.tStart = t  # local t and not account for scr refresh
                        answer_box.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(answer_box, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'answer_box.started')
                        # update status
                        answer_box.status = STARTED
                        answer_box.setAutoDraw(True)
                    
                    # if answer_box is active this frame...
                    if answer_box.status == STARTED:
                        # update params
                        pass
                    
                    # *go_on_btn* updates
                    
                    # if go_on_btn is starting this frame...
                    if go_on_btn.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        go_on_btn.frameNStart = frameN  # exact frame index
                        go_on_btn.tStart = t  # local t and not account for scr refresh
                        go_on_btn.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(go_on_btn, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'go_on_btn.started')
                        # update status
                        go_on_btn.status = STARTED
                        go_on_btn.setAutoDraw(True)
                    
                    # if go_on_btn is active this frame...
                    if go_on_btn.status == STARTED:
                        # update params
                        pass
                    
                    # *go_on_txt* updates
                    
                    # if go_on_txt is starting this frame...
                    if go_on_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        go_on_txt.frameNStart = frameN  # exact frame index
                        go_on_txt.tStart = t  # local t and not account for scr refresh
                        go_on_txt.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(go_on_txt, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'go_on_txt.started')
                        # update status
                        go_on_txt.status = STARTED
                        go_on_txt.setAutoDraw(True)
                    
                    # if go_on_txt is active this frame...
                    if go_on_txt.status == STARTED:
                        # update params
                        pass
                    # *go_on_mse* updates
                    
                    # if go_on_mse is starting this frame...
                    if go_on_mse.status == NOT_STARTED and answer_box.getText() != "":
                        # keep track of start time/frame for later
                        go_on_mse.frameNStart = frameN  # exact frame index
                        go_on_mse.tStart = t  # local t and not account for scr refresh
                        go_on_mse.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(go_on_mse, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.addData('go_on_mse.started', t)
                        # update status
                        go_on_mse.status = STARTED
                        go_on_mse.mouseClock.reset()
                        prevButtonState = go_on_mse.getPressed()  # if button is down already this ISN'T a new click
                    if go_on_mse.status == STARTED:  # only update if started and not finished!
                        buttons = go_on_mse.getPressed()
                        if buttons != prevButtonState:  # button state changed?
                            prevButtonState = buttons
                            if sum(buttons) > 0:  # state changed to a new click
                                # check if the mouse was inside our 'clickable' objects
                                gotValidClick = False
                                clickableList = environmenttools.getFromNames(go_on_btn, namespace=locals())
                                for obj in clickableList:
                                    # is this object clicked on?
                                    if obj.contains(go_on_mse):
                                        gotValidClick = True
                                        go_on_mse.clicked_name.append(obj.name)
                                x, y = go_on_mse.getPos()
                                go_on_mse.x.append(x)
                                go_on_mse.y.append(y)
                                buttons = go_on_mse.getPressed()
                                go_on_mse.leftButton.append(buttons[0])
                                go_on_mse.midButton.append(buttons[1])
                                go_on_mse.rightButton.append(buttons[2])
                                go_on_mse.time.append(go_on_mse.mouseClock.getTime())
                                if gotValidClick:
                                    continueRoutine = False  # end routine on response
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in t_AUTanswerComponents:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "t_AUTanswer" ---
                for thisComponent in t_AUTanswerComponents:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                thisExp.addData('t_AUTanswer.stopped', globalClock.getTime(format='float'))
                # Run 'End Routine' code from AUTanswer_code
                ##  marker: AUTanswer STOP  ##
                Marker.setData(55); core.wait(0.1); Marker.setData(0)
                AUTitem_loop.addData('answer_box.text',answer_box.text)
                # store data for AUTitem_loop (TrialHandler)
                AUTitem_loop.addData('go_on_mse.x', go_on_mse.x)
                AUTitem_loop.addData('go_on_mse.y', go_on_mse.y)
                AUTitem_loop.addData('go_on_mse.leftButton', go_on_mse.leftButton)
                AUTitem_loop.addData('go_on_mse.midButton', go_on_mse.midButton)
                AUTitem_loop.addData('go_on_mse.rightButton', go_on_mse.rightButton)
                AUTitem_loop.addData('go_on_mse.time', go_on_mse.time)
                AUTitem_loop.addData('go_on_mse.clicked_name', go_on_mse.clicked_name)
                # the Routine "t_AUTanswer" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
                
                # --- Prepare to start Routine "AUT_rating" ---
                continueRoutine = True
                # update component parameters for each repeat
                thisExp.addData('AUT_rating.started', globalClock.getTime(format='float'))
                VASrating.reset()
                # create starting attributes for r_end_key
                r_end_key.keys = []
                r_end_key.rt = []
                _r_end_key_allKeys = []
                # setup some python lists for storing info about the r1_mouse
                r1_mouse.clicked_name = []
                gotValidClick = False  # until a click is received
                # keep track of which components have finished
                AUT_ratingComponents = [r_header_txt, VASrating, r_end_key, r1_mouse, l_label, r_label, r1_btn, rating_hint]
                for thisComponent in AUT_ratingComponents:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "AUT_rating" ---
                routineForceEnded = not continueRoutine
                while continueRoutine:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *r_header_txt* updates
                    
                    # if r_header_txt is starting this frame...
                    if r_header_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        r_header_txt.frameNStart = frameN  # exact frame index
                        r_header_txt.tStart = t  # local t and not account for scr refresh
                        r_header_txt.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(r_header_txt, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'r_header_txt.started')
                        # update status
                        r_header_txt.status = STARTED
                        r_header_txt.setAutoDraw(True)
                    
                    # if r_header_txt is active this frame...
                    if r_header_txt.status == STARTED:
                        # update params
                        pass
                    
                    # *VASrating* updates
                    
                    # if VASrating is starting this frame...
                    if VASrating.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        VASrating.frameNStart = frameN  # exact frame index
                        VASrating.tStart = t  # local t and not account for scr refresh
                        VASrating.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(VASrating, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'VASrating.started')
                        # update status
                        VASrating.status = STARTED
                        VASrating.setAutoDraw(True)
                    
                    # if VASrating is active this frame...
                    if VASrating.status == STARTED:
                        # update params
                        pass
                    
                    # *r_end_key* updates
                    waitOnFlip = False
                    
                    # if r_end_key is starting this frame...
                    if r_end_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        r_end_key.frameNStart = frameN  # exact frame index
                        r_end_key.tStart = t  # local t and not account for scr refresh
                        r_end_key.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(r_end_key, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        r_end_key.status = STARTED
                        # keyboard checking is just starting
                        waitOnFlip = True
                        win.callOnFlip(r_end_key.clock.reset)  # t=0 on next screen flip
                        win.callOnFlip(r_end_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
                    if r_end_key.status == STARTED and not waitOnFlip:
                        theseKeys = r_end_key.getKeys(keyList=['w','return'], ignoreKeys=["escape"], waitRelease=False)
                        _r_end_key_allKeys.extend(theseKeys)
                        if len(_r_end_key_allKeys):
                            r_end_key.keys = _r_end_key_allKeys[-1].name  # just the last key pressed
                            r_end_key.rt = _r_end_key_allKeys[-1].rt
                            r_end_key.duration = _r_end_key_allKeys[-1].duration
                            # a response ends the routine
                            continueRoutine = False
                    # *r1_mouse* updates
                    
                    # if r1_mouse is starting this frame...
                    if r1_mouse.status == NOT_STARTED and VASrating.getMouseResponses() != None:
                        # keep track of start time/frame for later
                        r1_mouse.frameNStart = frameN  # exact frame index
                        r1_mouse.tStart = t  # local t and not account for scr refresh
                        r1_mouse.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(r1_mouse, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.addData('r1_mouse.started', t)
                        # update status
                        r1_mouse.status = STARTED
                        r1_mouse.mouseClock.reset()
                        prevButtonState = r1_mouse.getPressed()  # if button is down already this ISN'T a new click
                    if r1_mouse.status == STARTED:  # only update if started and not finished!
                        buttons = r1_mouse.getPressed()
                        if buttons != prevButtonState:  # button state changed?
                            prevButtonState = buttons
                            if sum(buttons) > 0:  # state changed to a new click
                                # check if the mouse was inside our 'clickable' objects
                                gotValidClick = False
                                clickableList = environmenttools.getFromNames(r1_btn, namespace=locals())
                                for obj in clickableList:
                                    # is this object clicked on?
                                    if obj.contains(r1_mouse):
                                        gotValidClick = True
                                        r1_mouse.clicked_name.append(obj.name)
                                if gotValidClick:  
                                    continueRoutine = False  # end routine on response
                    
                    # *l_label* updates
                    
                    # if l_label is starting this frame...
                    if l_label.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        l_label.frameNStart = frameN  # exact frame index
                        l_label.tStart = t  # local t and not account for scr refresh
                        l_label.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(l_label, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'l_label.started')
                        # update status
                        l_label.status = STARTED
                        l_label.setAutoDraw(True)
                    
                    # if l_label is active this frame...
                    if l_label.status == STARTED:
                        # update params
                        pass
                    
                    # *r_label* updates
                    
                    # if r_label is starting this frame...
                    if r_label.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        r_label.frameNStart = frameN  # exact frame index
                        r_label.tStart = t  # local t and not account for scr refresh
                        r_label.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(r_label, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'r_label.started')
                        # update status
                        r_label.status = STARTED
                        r_label.setAutoDraw(True)
                    
                    # if r_label is active this frame...
                    if r_label.status == STARTED:
                        # update params
                        pass
                    
                    # *r1_btn* updates
                    
                    # if r1_btn is starting this frame...
                    if r1_btn.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        r1_btn.frameNStart = frameN  # exact frame index
                        r1_btn.tStart = t  # local t and not account for scr refresh
                        r1_btn.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(r1_btn, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'r1_btn.started')
                        # update status
                        r1_btn.status = STARTED
                        r1_btn.setAutoDraw(True)
                    
                    # if r1_btn is active this frame...
                    if r1_btn.status == STARTED:
                        # update params
                        pass
                    
                    # *rating_hint* updates
                    
                    # if rating_hint is starting this frame...
                    if rating_hint.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        rating_hint.frameNStart = frameN  # exact frame index
                        rating_hint.tStart = t  # local t and not account for scr refresh
                        rating_hint.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(rating_hint, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'rating_hint.started')
                        # update status
                        rating_hint.status = STARTED
                        rating_hint.setAutoDraw(True)
                    
                    # if rating_hint is active this frame...
                    if rating_hint.status == STARTED:
                        # update params
                        pass
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in AUT_ratingComponents:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "AUT_rating" ---
                for thisComponent in AUT_ratingComponents:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                thisExp.addData('AUT_rating.stopped', globalClock.getTime(format='float'))
                # Run 'End Routine' code from AUTrating_code
                ##  marker: AUTrating STOP  ##
                Marker.setData(57); core.wait(0.1); Marker.setData(0)
                AUTitem_loop.addData('VASrating.response', VASrating.getRating())
                AUTitem_loop.addData('VASrating.rt', VASrating.getRT())
                # check responses
                if r_end_key.keys in ['', [], None]:  # No response was made
                    r_end_key.keys = None
                AUTitem_loop.addData('r_end_key.keys',r_end_key.keys)
                if r_end_key.keys != None:  # we had a response
                    AUTitem_loop.addData('r_end_key.rt', r_end_key.rt)
                    AUTitem_loop.addData('r_end_key.duration', r_end_key.duration)
                # store data for AUTitem_loop (TrialHandler)
                x, y = r1_mouse.getPos()
                buttons = r1_mouse.getPressed()
                if sum(buttons):
                    # check if the mouse was inside our 'clickable' objects
                    gotValidClick = False
                    clickableList = environmenttools.getFromNames(r1_btn, namespace=locals())
                    for obj in clickableList:
                        # is this object clicked on?
                        if obj.contains(r1_mouse):
                            gotValidClick = True
                            r1_mouse.clicked_name.append(obj.name)
                AUTitem_loop.addData('r1_mouse.x', x)
                AUTitem_loop.addData('r1_mouse.y', y)
                AUTitem_loop.addData('r1_mouse.leftButton', buttons[0])
                AUTitem_loop.addData('r1_mouse.midButton', buttons[1])
                AUTitem_loop.addData('r1_mouse.rightButton', buttons[2])
                if len(r1_mouse.clicked_name):
                    AUTitem_loop.addData('r1_mouse.clicked_name', r1_mouse.clicked_name[0])
                # the Routine "AUT_rating" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
                thisExp.nextEntry()
                
                if thisSession is not None:
                    # if running in a Session with a Liaison client, send data up to now
                    thisSession.sendExperimentData()
            # completed 1 repeats of 'AUTitem_loop'
            
            thisExp.nextEntry()
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
        # completed AUT_run repeats of 'AUT_trials'
        
        
        # set up handler to look after randomisation of conditions etc
        TTCT_trials = data.TrialHandler(nReps=TTCT_run, method='random', 
            extraInfo=expInfo, originPath=-1,
            trialList=[None],
            seed=None, name='TTCT_trials')
        thisExp.addLoop(TTCT_trials)  # add the loop to the experiment
        thisTTCT_trial = TTCT_trials.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTTCT_trial.rgb)
        if thisTTCT_trial != None:
            for paramName in thisTTCT_trial:
                globals()[paramName] = thisTTCT_trial[paramName]
        
        for thisTTCT_trial in TTCT_trials:
            currentLoop = TTCT_trials
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
            )
            # abbreviate parameter names if possible (e.g. rgb = thisTTCT_trial.rgb)
            if thisTTCT_trial != None:
                for paramName in thisTTCT_trial:
                    globals()[paramName] = thisTTCT_trial[paramName]
            
            # --- Prepare to start Routine "b2TTCTonline" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('b2TTCTonline.started', globalClock.getTime(format='float'))
            # Run 'Begin Routine' code from b2_TTCT_code
            ##  marker: TTCT START  ##
            Marker.setData(61); core.wait(0.1); Marker.setData(0)
            
            ##  select current TTCTitem
            TTCTitem_presented = False      #flag for 'wait'-routine
            TTCTitem = TTCTitems_block1[presented_TTCTitems]
            TTCT_trials.addData('TTCTitem', TTCTitem)  #’trials’=current loop
            
            
            ##  handle (some) windows
            win32gui.ShowWindow(runParadigmaWindow, win32con.SW_MINIMIZE)       #Minimize "running paradigma"(-PsychoPy) window
            
            
            ##  web interaction
            #url = f"https://webpsy2.uni-graz.at/ges1www/crea_draw/?studyId=999&participantId=77"
            url = f"https://webpsy2.uni-graz.at/ges1www/crea_draw/?studyId={study_ID}_{TTCTitem}&participantId={participant_ID}"
            
            try:
                #print("browser.get(url)")
                browser.get(url)                            #Navigate to website
                browser.maximize_window()                   #Maximize the browser window
                    
                # ##Wait for alert and handle browser closure
                wait_for_alert_and_close(browser)      #no specific 'time-out'-param (default = 60s)
                #wait_for_alert_and_close(browser, 5)    #set a 'time-out'-param
                
            except Exception as e:
                print(f"An error occurred: {e}")
                #browser.quit()
            
            core.wait(.1)
            
            # keep track of which components have finished
            b2TTCTonlineComponents = []
            for thisComponent in b2TTCTonlineComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "b2TTCTonline" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in b2TTCTonlineComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "b2TTCTonline" ---
            for thisComponent in b2TTCTonlineComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('b2TTCTonline.stopped', globalClock.getTime(format='float'))
            # Run 'End Routine' code from b2_TTCT_code
            ##  marker: TTCT STOP  ##
            Marker.setData(63); core.wait(0.1); Marker.setData(0)
            
            presented_TTCTitems += 1
            #print(f'{presented_TTCTitems=}')
            TTCTitem_presented = True
            
            ##  maximize >runParadigma<-window
            win32gui.ShowWindow(runParadigmaWindow, win32con.SW_MAXIMIZE)           #maximize "running paradigma"(-PsychoPy) window
            
            
            ##  Close the browser window and wait
            #browser.quit()
            core.wait(0.1)
            
            """
            ##  maximize >runParadigma<-window
            win32gui.ShowWindow(runParadigmaWindow, win32con.SW_MAXIMIZE)           #maximize "running paradigma"(-PsychoPy) window
            """
            
            
            # the Routine "b2TTCTonline" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "wait" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('wait.started', globalClock.getTime(format='float'))
            # Run 'Begin Routine' code from wait_code
            if TTCTitem_presented:
                continueRoutine = False
            # keep track of which components have finished
            waitComponents = []
            for thisComponent in waitComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "wait" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in waitComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "wait" ---
            for thisComponent in waitComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('wait.stopped', globalClock.getTime(format='float'))
            # the Routine "wait" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "TTCT_rating" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('TTCT_rating.started', globalClock.getTime(format='float'))
            TTCT_VASrating.reset()
            # create starting attributes for TTCT_r_end_key
            TTCT_r_end_key.keys = []
            TTCT_r_end_key.rt = []
            _TTCT_r_end_key_allKeys = []
            # setup some python lists for storing info about the TTCT_r1_mse
            TTCT_r1_mse.clicked_name = []
            gotValidClick = False  # until a click is received
            # keep track of which components have finished
            TTCT_ratingComponents = [TTCT_r_header_txt, TTCT_VASrating, TTCT_r_end_key, TTCT_r1_mse, TTCT_l_label, TTCT_r_label, TTCT_r1_btn, TTCT_rating_hint]
            for thisComponent in TTCT_ratingComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "TTCT_rating" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *TTCT_r_header_txt* updates
                
                # if TTCT_r_header_txt is starting this frame...
                if TTCT_r_header_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    TTCT_r_header_txt.frameNStart = frameN  # exact frame index
                    TTCT_r_header_txt.tStart = t  # local t and not account for scr refresh
                    TTCT_r_header_txt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(TTCT_r_header_txt, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'TTCT_r_header_txt.started')
                    # update status
                    TTCT_r_header_txt.status = STARTED
                    TTCT_r_header_txt.setAutoDraw(True)
                
                # if TTCT_r_header_txt is active this frame...
                if TTCT_r_header_txt.status == STARTED:
                    # update params
                    pass
                
                # *TTCT_VASrating* updates
                
                # if TTCT_VASrating is starting this frame...
                if TTCT_VASrating.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    TTCT_VASrating.frameNStart = frameN  # exact frame index
                    TTCT_VASrating.tStart = t  # local t and not account for scr refresh
                    TTCT_VASrating.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(TTCT_VASrating, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'TTCT_VASrating.started')
                    # update status
                    TTCT_VASrating.status = STARTED
                    TTCT_VASrating.setAutoDraw(True)
                
                # if TTCT_VASrating is active this frame...
                if TTCT_VASrating.status == STARTED:
                    # update params
                    pass
                
                # *TTCT_r_end_key* updates
                waitOnFlip = False
                
                # if TTCT_r_end_key is starting this frame...
                if TTCT_r_end_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    TTCT_r_end_key.frameNStart = frameN  # exact frame index
                    TTCT_r_end_key.tStart = t  # local t and not account for scr refresh
                    TTCT_r_end_key.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(TTCT_r_end_key, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    TTCT_r_end_key.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(TTCT_r_end_key.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(TTCT_r_end_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if TTCT_r_end_key.status == STARTED and not waitOnFlip:
                    theseKeys = TTCT_r_end_key.getKeys(keyList=['w','return'], ignoreKeys=["escape"], waitRelease=False)
                    _TTCT_r_end_key_allKeys.extend(theseKeys)
                    if len(_TTCT_r_end_key_allKeys):
                        TTCT_r_end_key.keys = _TTCT_r_end_key_allKeys[-1].name  # just the last key pressed
                        TTCT_r_end_key.rt = _TTCT_r_end_key_allKeys[-1].rt
                        TTCT_r_end_key.duration = _TTCT_r_end_key_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                # *TTCT_r1_mse* updates
                
                # if TTCT_r1_mse is starting this frame...
                if TTCT_r1_mse.status == NOT_STARTED and TTCT_VASrating.getMouseResponses() != None:
                    # keep track of start time/frame for later
                    TTCT_r1_mse.frameNStart = frameN  # exact frame index
                    TTCT_r1_mse.tStart = t  # local t and not account for scr refresh
                    TTCT_r1_mse.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(TTCT_r1_mse, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.addData('TTCT_r1_mse.started', t)
                    # update status
                    TTCT_r1_mse.status = STARTED
                    TTCT_r1_mse.mouseClock.reset()
                    prevButtonState = TTCT_r1_mse.getPressed()  # if button is down already this ISN'T a new click
                if TTCT_r1_mse.status == STARTED:  # only update if started and not finished!
                    buttons = TTCT_r1_mse.getPressed()
                    if buttons != prevButtonState:  # button state changed?
                        prevButtonState = buttons
                        if sum(buttons) > 0:  # state changed to a new click
                            # check if the mouse was inside our 'clickable' objects
                            gotValidClick = False
                            clickableList = environmenttools.getFromNames(TTCT_r1_btn, namespace=locals())
                            for obj in clickableList:
                                # is this object clicked on?
                                if obj.contains(TTCT_r1_mse):
                                    gotValidClick = True
                                    TTCT_r1_mse.clicked_name.append(obj.name)
                            if gotValidClick:  
                                continueRoutine = False  # end routine on response
                
                # *TTCT_l_label* updates
                
                # if TTCT_l_label is starting this frame...
                if TTCT_l_label.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    TTCT_l_label.frameNStart = frameN  # exact frame index
                    TTCT_l_label.tStart = t  # local t and not account for scr refresh
                    TTCT_l_label.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(TTCT_l_label, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'TTCT_l_label.started')
                    # update status
                    TTCT_l_label.status = STARTED
                    TTCT_l_label.setAutoDraw(True)
                
                # if TTCT_l_label is active this frame...
                if TTCT_l_label.status == STARTED:
                    # update params
                    pass
                
                # *TTCT_r_label* updates
                
                # if TTCT_r_label is starting this frame...
                if TTCT_r_label.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    TTCT_r_label.frameNStart = frameN  # exact frame index
                    TTCT_r_label.tStart = t  # local t and not account for scr refresh
                    TTCT_r_label.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(TTCT_r_label, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'TTCT_r_label.started')
                    # update status
                    TTCT_r_label.status = STARTED
                    TTCT_r_label.setAutoDraw(True)
                
                # if TTCT_r_label is active this frame...
                if TTCT_r_label.status == STARTED:
                    # update params
                    pass
                
                # *TTCT_r1_btn* updates
                
                # if TTCT_r1_btn is starting this frame...
                if TTCT_r1_btn.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    TTCT_r1_btn.frameNStart = frameN  # exact frame index
                    TTCT_r1_btn.tStart = t  # local t and not account for scr refresh
                    TTCT_r1_btn.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(TTCT_r1_btn, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'TTCT_r1_btn.started')
                    # update status
                    TTCT_r1_btn.status = STARTED
                    TTCT_r1_btn.setAutoDraw(True)
                
                # if TTCT_r1_btn is active this frame...
                if TTCT_r1_btn.status == STARTED:
                    # update params
                    pass
                
                # *TTCT_rating_hint* updates
                
                # if TTCT_rating_hint is starting this frame...
                if TTCT_rating_hint.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    TTCT_rating_hint.frameNStart = frameN  # exact frame index
                    TTCT_rating_hint.tStart = t  # local t and not account for scr refresh
                    TTCT_rating_hint.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(TTCT_rating_hint, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'TTCT_rating_hint.started')
                    # update status
                    TTCT_rating_hint.status = STARTED
                    TTCT_rating_hint.setAutoDraw(True)
                
                # if TTCT_rating_hint is active this frame...
                if TTCT_rating_hint.status == STARTED:
                    # update params
                    pass
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in TTCT_ratingComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "TTCT_rating" ---
            for thisComponent in TTCT_ratingComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('TTCT_rating.stopped', globalClock.getTime(format='float'))
            TTCT_trials.addData('TTCT_VASrating.response', TTCT_VASrating.getRating())
            TTCT_trials.addData('TTCT_VASrating.rt', TTCT_VASrating.getRT())
            TTCT_trials.addData('TTCT_VASrating.history', TTCT_VASrating.getHistory())
            # check responses
            if TTCT_r_end_key.keys in ['', [], None]:  # No response was made
                TTCT_r_end_key.keys = None
            TTCT_trials.addData('TTCT_r_end_key.keys',TTCT_r_end_key.keys)
            if TTCT_r_end_key.keys != None:  # we had a response
                TTCT_trials.addData('TTCT_r_end_key.rt', TTCT_r_end_key.rt)
                TTCT_trials.addData('TTCT_r_end_key.duration', TTCT_r_end_key.duration)
            # store data for TTCT_trials (TrialHandler)
            x, y = TTCT_r1_mse.getPos()
            buttons = TTCT_r1_mse.getPressed()
            if sum(buttons):
                # check if the mouse was inside our 'clickable' objects
                gotValidClick = False
                clickableList = environmenttools.getFromNames(TTCT_r1_btn, namespace=locals())
                for obj in clickableList:
                    # is this object clicked on?
                    if obj.contains(TTCT_r1_mse):
                        gotValidClick = True
                        TTCT_r1_mse.clicked_name.append(obj.name)
            TTCT_trials.addData('TTCT_r1_mse.x', x)
            TTCT_trials.addData('TTCT_r1_mse.y', y)
            TTCT_trials.addData('TTCT_r1_mse.leftButton', buttons[0])
            TTCT_trials.addData('TTCT_r1_mse.midButton', buttons[1])
            TTCT_trials.addData('TTCT_r1_mse.rightButton', buttons[2])
            if len(TTCT_r1_mse.clicked_name):
                TTCT_trials.addData('TTCT_r1_mse.clicked_name', TTCT_r1_mse.clicked_name[0])
            # Run 'End Routine' code from TTCTrating_code
            ##  marker: TTCTrating STOP  ##
            Marker.setData(65); core.wait(0.1); Marker.setData(0)
            # the Routine "TTCT_rating" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
        # completed TTCT_run repeats of 'TTCT_trials'
        
        
        # set up handler to look after randomisation of conditions etc
        NB1min_loop = data.TrialHandler(nReps=go_NB1min, method='sequential', 
            extraInfo=expInfo, originPath=-1,
            trialList=[None],
            seed=None, name='NB1min_loop')
        thisExp.addLoop(NB1min_loop)  # add the loop to the experiment
        thisNB1min_loop = NB1min_loop.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisNB1min_loop.rgb)
        if thisNB1min_loop != None:
            for paramName in thisNB1min_loop:
                globals()[paramName] = thisNB1min_loop[paramName]
        
        for thisNB1min_loop in NB1min_loop:
            currentLoop = NB1min_loop
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
            )
            # abbreviate parameter names if possible (e.g. rgb = thisNB1min_loop.rgb)
            if thisNB1min_loop != None:
                for paramName in thisNB1min_loop:
                    globals()[paramName] = thisNB1min_loop[paramName]
            
            # --- Prepare to start Routine "NB_1min" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('NB_1min.started', globalClock.getTime(format='float'))
            # Run 'Begin Routine' code from NB1min_code
            ##  marker: NB1min START  ##
            Marker.setData(25); core.wait(0.1); Marker.setData(0)
            
            # keep track of which components have finished
            NB_1minComponents = [NB1min_txt, NB1m_countdown]
            for thisComponent in NB_1minComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "NB_1min" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *NB1min_txt* updates
                
                # if NB1min_txt is starting this frame...
                if NB1min_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    NB1min_txt.frameNStart = frameN  # exact frame index
                    NB1min_txt.tStart = t  # local t and not account for scr refresh
                    NB1min_txt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(NB1min_txt, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'NB1min_txt.started')
                    # update status
                    NB1min_txt.status = STARTED
                    NB1min_txt.setAutoDraw(True)
                
                # if NB1min_txt is active this frame...
                if NB1min_txt.status == STARTED:
                    # update params
                    pass
                
                # if NB1min_txt is stopping this frame...
                if NB1min_txt.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > NB1min_txt.tStartRefresh + crea_breathing_dur-frameTolerance:
                        # keep track of stop time/frame for later
                        NB1min_txt.tStop = t  # not accounting for scr refresh
                        NB1min_txt.tStopRefresh = tThisFlipGlobal  # on global time
                        NB1min_txt.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'NB1min_txt.stopped')
                        # update status
                        NB1min_txt.status = FINISHED
                        NB1min_txt.setAutoDraw(False)
                
                # *NB1m_countdown* updates
                
                # if NB1m_countdown is starting this frame...
                if NB1m_countdown.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    NB1m_countdown.frameNStart = frameN  # exact frame index
                    NB1m_countdown.tStart = t  # local t and not account for scr refresh
                    NB1m_countdown.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(NB1m_countdown, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'NB1m_countdown.started')
                    # update status
                    NB1m_countdown.status = STARTED
                    NB1m_countdown.setAutoDraw(True)
                
                # if NB1m_countdown is active this frame...
                if NB1m_countdown.status == STARTED:
                    # update params
                    NB1m_countdown.setText(str(crea_breathing_dur-int(t)), log=False)
                
                # if NB1m_countdown is stopping this frame...
                if NB1m_countdown.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > NB1m_countdown.tStartRefresh + crea_breathing_dur-frameTolerance:
                        # keep track of stop time/frame for later
                        NB1m_countdown.tStop = t  # not accounting for scr refresh
                        NB1m_countdown.tStopRefresh = tThisFlipGlobal  # on global time
                        NB1m_countdown.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'NB1m_countdown.stopped')
                        # update status
                        NB1m_countdown.status = FINISHED
                        NB1m_countdown.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in NB_1minComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "NB_1min" ---
            for thisComponent in NB_1minComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('NB_1min.stopped', globalClock.getTime(format='float'))
            # Run 'End Routine' code from NB1min_code
            ##  marker: NB1min STOP  ##
            Marker.setData(27); core.wait(0.1); Marker.setData(0)
            
            # the Routine "NB_1min" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
        # completed go_NB1min repeats of 'NB1min_loop'
        
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'crea_blocks'
    
    
    # set up handler to look after randomisation of conditions etc
    thx_byp = data.TrialHandler(nReps=disp_thx, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='thx_byp')
    thisExp.addLoop(thx_byp)  # add the loop to the experiment
    thisThx_byp = thx_byp.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisThx_byp.rgb)
    if thisThx_byp != None:
        for paramName in thisThx_byp:
            globals()[paramName] = thisThx_byp[paramName]
    
    for thisThx_byp in thx_byp:
        currentLoop = thx_byp
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisThx_byp.rgb)
        if thisThx_byp != None:
            for paramName in thisThx_byp:
                globals()[paramName] = thisThx_byp[paramName]
        
        # --- Prepare to start Routine "thx" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('thx.started', globalClock.getTime(format='float'))
        # create starting attributes for thx_key
        thx_key.keys = []
        thx_key.rt = []
        _thx_key_allKeys = []
        # keep track of which components have finished
        thxComponents = [thx_txt, thx_key, thx_hint]
        for thisComponent in thxComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "thx" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 5.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *thx_txt* updates
            
            # if thx_txt is starting this frame...
            if thx_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                thx_txt.frameNStart = frameN  # exact frame index
                thx_txt.tStart = t  # local t and not account for scr refresh
                thx_txt.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(thx_txt, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'thx_txt.started')
                # update status
                thx_txt.status = STARTED
                thx_txt.setAutoDraw(True)
            
            # if thx_txt is active this frame...
            if thx_txt.status == STARTED:
                # update params
                pass
            
            # if thx_txt is stopping this frame...
            if thx_txt.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > thx_txt.tStartRefresh + 5-frameTolerance:
                    # keep track of stop time/frame for later
                    thx_txt.tStop = t  # not accounting for scr refresh
                    thx_txt.tStopRefresh = tThisFlipGlobal  # on global time
                    thx_txt.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'thx_txt.stopped')
                    # update status
                    thx_txt.status = FINISHED
                    thx_txt.setAutoDraw(False)
            
            # *thx_key* updates
            waitOnFlip = False
            
            # if thx_key is starting this frame...
            if thx_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                thx_key.frameNStart = frameN  # exact frame index
                thx_key.tStart = t  # local t and not account for scr refresh
                thx_key.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(thx_key, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'thx_key.started')
                # update status
                thx_key.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(thx_key.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(thx_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if thx_key is stopping this frame...
            if thx_key.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > thx_key.tStartRefresh + 5-frameTolerance:
                    # keep track of stop time/frame for later
                    thx_key.tStop = t  # not accounting for scr refresh
                    thx_key.tStopRefresh = tThisFlipGlobal  # on global time
                    thx_key.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'thx_key.stopped')
                    # update status
                    thx_key.status = FINISHED
                    thx_key.status = FINISHED
            if thx_key.status == STARTED and not waitOnFlip:
                theseKeys = thx_key.getKeys(keyList=['return','space'], ignoreKeys=["escape"], waitRelease=False)
                _thx_key_allKeys.extend(theseKeys)
                if len(_thx_key_allKeys):
                    thx_key.keys = _thx_key_allKeys[-1].name  # just the last key pressed
                    thx_key.rt = _thx_key_allKeys[-1].rt
                    thx_key.duration = _thx_key_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *thx_hint* updates
            
            # if thx_hint is starting this frame...
            if thx_hint.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                thx_hint.frameNStart = frameN  # exact frame index
                thx_hint.tStart = t  # local t and not account for scr refresh
                thx_hint.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(thx_hint, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'thx_hint.started')
                # update status
                thx_hint.status = STARTED
                thx_hint.setAutoDraw(True)
            
            # if thx_hint is active this frame...
            if thx_hint.status == STARTED:
                # update params
                pass
            
            # if thx_hint is stopping this frame...
            if thx_hint.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > thx_hint.tStartRefresh + 5-frameTolerance:
                    # keep track of stop time/frame for later
                    thx_hint.tStop = t  # not accounting for scr refresh
                    thx_hint.tStopRefresh = tThisFlipGlobal  # on global time
                    thx_hint.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'thx_hint.stopped')
                    # update status
                    thx_hint.status = FINISHED
                    thx_hint.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in thxComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "thx" ---
        for thisComponent in thxComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('thx.stopped', globalClock.getTime(format='float'))
        # check responses
        if thx_key.keys in ['', [], None]:  # No response was made
            thx_key.keys = None
        thx_byp.addData('thx_key.keys',thx_key.keys)
        if thx_key.keys != None:  # we had a response
            thx_byp.addData('thx_key.rt', thx_key.rt)
            thx_byp.addData('thx_key.duration', thx_key.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-5.000000)
    # completed disp_thx repeats of 'thx_byp'
    
    # Run 'End Experiment' code from set_things
    ##  marker: NB STOP  ##
    Marker.setData(203); core.wait(0.1); Marker.setData(0)
    # Run 'End Experiment' code from b2_TTCT_code
    ##  Maximize 'PsychoPy-Builder' Window
    #win32gui.ShowWindow(PsychoPyBuilderWin, win32con.SW_MAXIMIZE)
    #core.wait(0.1)
    
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # shut down eyetracker, if there is one
    if deviceManager.getDevice('eyetracker') is not None:
        deviceManager.removeDevice('eyetracker')
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    # shut down eyetracker, if there is one
    if deviceManager.getDevice('eyetracker') is not None:
        deviceManager.removeDevice('eyetracker')
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
