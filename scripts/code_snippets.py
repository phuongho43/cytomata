

def rescale(aa):
    return (aa - min(aa)) / (max(aa) - min(aa))


def sse(aa, bb):
    return np.sum((aa - bb)**2)


def approx_half_life(t, y, phase='fall'):
    """Approximate half life of reaction process using cubic spline interpolation."""
    t = np.array(t)
    y = np.array(y)
    if phase == 'rise':
        tp = t[:y.argmax()]
        yp = y[:y.argmax()]
    elif phase == 'fall':
        tp = t[y.argmax():]
        yp = y[y.argmax():]
    y_half = (np.max(y) - np.min(y))/2
    yf = interp1d(tp, yp, 'cubic')
    ti = np.arange(tp[0], tp[-1], 1)
    yi = yf(ti)
    idx = np.argmin((yi - y_half)**2)
    t_half = ti[idx]
    return t_half


def imgs_to_mp4(imgs, vid_path, fps=10):
    for i, img in enumerate(imgs):
        img = img_as_ubyte(img)
        if i == 0:
            height, width = imgs[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            vid_path = vid_path + '.mp4' if not vid_path.endswith('.mp4') else vid_path
            video = cv2.VideoWriter(vid_path, fourcc, fps, (width, height))
        video.write(img)
    cv2.destroyAllWindows()
    video.release()
 

def process_translo(nucleus_dir, translo_dir, results_dir):
    setup_dirs(os.path.join(results_dir, 'imgs'))
    t = []
    yc = []
    yn = []
    nu_imgs = []
    tr_imgs = []
    nu_imgfs = list_img_files(nucleus_dir)
    tr_imgfs = list_img_files(translo_dir)
    n_imgs = min(len(nu_imgfs), len(tr_imgfs))
    nu_tval = None
    tr_tval = None
    for i, (nu_imgf, tr_imgf) in enumerate(tqdm(zip(nu_imgfs, tr_imgfs), total=n_imgs)):
        nu_img, nu_tval = preprocess_img(nu_imgf, tval=nu_tval)
        tr_img, tr_tval = preprocess_img(tr_imgf, tval=tr_tval)
        nucl = segment_object(nu_img, er=7, cb=5)
        cell = segment_object(tr_img, er=7)
        cell = np.logical_or(cell, nucl)
        cyto = np.logical_xor(cell, nucl)
        nroi = nucl*tr_img
        croi = cyto*tr_img
        nnz = nroi[nroi > 0]
        cnz = croi[croi > 0]
        nucl_int = np.median(nnz)
        cyto_int = np.median(cnz)
        fname = os.path.splitext(os.path.basename(tr_imgf))[0]
        t.append(np.float(fname))
        yn.append(nucl_int)
        yc.append(cyto_int)
        if i == 0:
            nucl_cmin = np.min(nu_img)
            nucl_cmax = 1.1*np.max(nu_img)
            cell_cmin = np.min(tr_img)
            cell_cmax = 1.1*np.max(tr_img)
        figh = nu_img.shape[0]/100
        figw = nu_img.shape[1]/100
        fig, ax = plt.subplots(figsize=(figw, figh))
        axim = ax.imshow(nu_img, cmap='turbo')
        axim.set_clim(nucl_cmin, nucl_cmax)
        ax.grid(False)
        ax.axis('off')
        fig.tight_layout(pad=0)
        fig.canvas.draw()
        cmimg = np.array(fig.canvas.renderer._renderer)
        nu_imgs.append(cmimg)
        plt.close(fig)
        figh = tr_img.shape[0]/100
        figw = tr_img.shape[1]/100
        fig, ax = plt.subplots(figsize=(figw, figh))
        axim = ax.imshow(tr_img, cmap='turbo')
        axim.set_clim(cell_cmin, cell_cmax)
        ax.grid(False)
        ax.axis('off')
        fig.tight_layout(pad=0)
        fig.canvas.draw()
        cmimg = np.array(fig.canvas.renderer._renderer)
        tr_imgs.append(cmimg)
        plt.close(fig)
        with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
            fig, ax = plt.subplots(figsize=(10,8))
            axim = ax.imshow(tr_img, cmap='turbo')
            axim.set_clim(cell_cmin, cell_cmax)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ax.contour(cyto, linewidths=1.0, colors='w')
                ax.contour(nucl, linewidths=0.2, colors='r')
            ax.grid(False)
            ax.axis('off')
            fig.tight_layout(pad=0)
            cb = fig.colorbar(axim, pad=0.01, format='%.4f',
                extend='both', extendrect=True, extendfrac=0.025)
            cb.outline.set_linewidth(0)
            fig.canvas.draw()
            fig.savefig(os.path.join(results_dir, 'imgs', fname + '.png'),
                dpi=100, bbox_inches='tight', transparent=True, pad_inches=0)
            plt.close(fig)
    data = np.column_stack((t, yc, yn))
    np.savetxt(os.path.join(results_dir, 'y.csv'),
        data, delimiter=',', header='t,yc,yn', comments='')
    y = np.column_stack((yc, yn))
    # plot(t, y, xlabel='Time (s)', ylabel='AU',
    #     labels=['Cytoplasm', 'Nucleus'], save_path=os.path.join(results_dir, 'plot.png'))
    mimwrite(os.path.join(results_dir, 'nucl.gif'), nu_imgs, fps=n_imgs//12)
    mimwrite(os.path.join(results_dir, 'cell.gif'), tr_imgs, fps=n_imgs//12)


def combine_translo(results_dir):
    with plt.style.context(('seaborn-whitegrid', custom_styles)):
        fig, ax = plt.subplots()
        combined_yc = pd.DataFrame()
        combined_yn = pd.DataFrame()
        for i, data_dir in enumerate(natsorted([x[1] for x in os.walk(results_dir)][0])):
            csv_fp = os.path.join(results_dir, data_dir, 'y.csv')
            data = pd.read_csv(csv_fp)
            t = data['t'].values
            yc = data['yc'].values
            yn = data['yn'].values
            ycf = interp1d(t, yc, fill_value='extrapolate')
            ynf = interp1d(t, yn, fill_value='extrapolate')
            t = np.arange(round(t[0]), round(t[-1]) + 1, 1)
            yc = pd.Series([ycf(ti) for ti in t], index=t, name=i)
            yn = pd.Series([ynf(ti) for ti in t], index=t, name=i)
            combined_yc = pd.concat([combined_yc, yc], axis=1)
            combined_yn = pd.concat([combined_yn, yn], axis=1)
            ax.plot(yc, color='#BBDEFB', linewidth=3)
            ax.plot(yn, color='#ffcdd2', linewidth=3)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('AU')
        fig.savefig(os.path.join(results_dir, 'yi.png'),
            dpi=100, bbox_inches='tight', transparent=False, pad_inches=0)
        plt.close(fig)
        yc_ave = combined_yc.mean(axis=1).rename('yc_ave')
        yc_std = combined_yc.std(axis=1).rename('yc_std')
        yc_sem = combined_yc.sem(axis=1).rename('yc_sem')
        yn_ave = combined_yn.mean(axis=1).rename('yn_ave')
        yn_std = combined_yn.std(axis=1).rename('yn_std')
        yn_sem = combined_yn.sem(axis=1).rename('yn_sem')
        data = pd.concat([yc_ave, yc_std, yc_sem, yn_ave, yn_std, yn_sem], axis=1).dropna()
        data.to_csv(os.path.join(results_dir, 'y_ave.csv'))
        yc_ave = data['yc_ave']
        yn_ave = data['yn_ave']
        yc_ci = data['yc_sem']*1.96
        yn_ci = data['yn_sem']*1.96
        fig, ax = plt.subplots()
        ax.plot(yc_ave, color='#1976D2', label='Ave Cytoplasmic')
        ax.plot(yn_ave, color='#d32f2f', label='Ave Nuclear')
        ax.fill_between(t, (yc_ave - yc_ci), (yc_ave + yc_ci), color='#1976D2', alpha=.1)
        ax.fill_between(t, (yn_ave - yn_ci), (yn_ave + yn_ci), color='#d32f2f', alpha=.1)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('AU')
        ax.legend(loc='best')
        fig.savefig(os.path.join(results_dir, 'ave.png'),
            dpi=100, bbox_inches='tight', transparent=False, pad_inches=0)
        plt.close(fig)



def combine_uy(root_dir, fold_change=True, plot_u=True):
    with plt.style.context(('seaborn-whitegrid', custom_styles)):
        if plot_u:
            fig, (ax0, ax) = plt.subplots(2, 1, sharex=True,
                figsize=(16, 10), gridspec_kw={'height_ratios': [1, 8]})
        else:
            fig, ax = plt.subplots(figsize=(10,8)) 
        combined_t = pd.DataFrame()
        combined_y = pd.DataFrame()
        combined_tu = pd.DataFrame()
        combined_u = pd.DataFrame()
        for i, data_dir in enumerate(natsorted([x[1] for x in os.walk(root_dir)][0])):
            y_csv = os.path.join(root_dir, data_dir, 'y.csv')
            y_data = pd.read_csv(y_csv)
            t = y_data['t'].values
            y = y_data['y'].values
            yf = interp1d(t, y, fill_value='extrapolate')
            t = np.arange(round(t[0]), round(t[-1]) + 1, 1)
            t = pd.Series(t, index=t, name=i)
            combined_t = pd.concat([combined_t, t], axis=1)
            y = pd.Series([yf(ti) for ti in t], index=t, name=i)
            if fold_change:
                y = y/np.mean(y[:5])
            combined_y = pd.concat([combined_y, y], axis=1)
            # ax.plot(y, color='#1976D2', alpha=0.5, marker='.', markersize=2, linewidth=0)
            u_csv = os.path.join(root_dir, data_dir, 'u.csv')
            if plot_u:
                u_data = pd.read_csv(u_csv)
                tu = u_data['t'].values
                tu = pd.Series(tu, index=tu, name=i)
                u = pd.Series(u_data['u'].values, index=tu, name=i)
                combined_tu = pd.concat([combined_tu, tu], axis=1)
                combined_u = pd.concat([combined_u, u], axis=1)
        t_ave = combined_t.mean(axis=1).rename('t')
        y_ave = combined_y.mean(axis=1).rename('y_ave')
        y_std = combined_y.std(axis=1).rename('y_std')
        y_sem = combined_y.sem(axis=1).rename('y_sem')
        if plot_u:
            tu_ave = combined_tu.mean(axis=1).rename('tu_ave')
            u_ave = combined_u.mean(axis=1).rename('u_ave')
            u_data = pd.concat([tu_ave, u_ave], axis=1).dropna()
            u_data.to_csv(os.path.join(root_dir, 'u_combined.csv'), index=False)
        y_data = pd.concat([t_ave, y_ave, y_std, y_sem], axis=1).dropna()
        y_data.to_csv(os.path.join(root_dir, 'y_combined.csv'), index=False)
        y_ave = y_data['y_ave']
        y_ci = y_data['y_sem']*1.96
        ax.fill_between(t, (y_ave - y_ci), (y_ave + y_ci), color='#1976D2', alpha=.2, label='95% CI')
        ax.plot(y_ave, color='#1976D2', label='Ave')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('AU')
        if fold_change:
            ax.set_ylabel('Fold Change')
        ax.legend(loc='best')
        if plot_u:
            ax0.plot(tu, u, color='#1976D2')
            ax0.set_yticks([0, 1])
            ax0.set_ylabel('BL')
        plot_name = 'y_combined.png'
        fig.savefig(os.path.join(root_dir, plot_name),
            dpi=100, bbox_inches='tight', transparent=False)
        plt.close(fig)


def process_ratio_fluo(fp0_img_dir, fp1_img_dir, results_dir):
    """Analyze PQR fluorescent intensities and generate figures."""
    setup_dirs(os.path.join(results_dir, 'fp0_imgs'))
    setup_dirs(os.path.join(results_dir, 'fp1_imgs'))
    y0 = []
    y1 = []
    imgfs0 = list_img_files(fp0_img_dir)
    imgfs1 = list_img_files(fp1_img_dir)
    for i, (imgf0, imgf1) in enumerate(tqdm(zip(imgfs0, imgfs1), total=len(imgfs0))):
        fname0 = os.path.splitext(os.path.basename(imgf0))[0]
        fname1 = os.path.splitext(os.path.basename(imgf1))[0]
        img0, tval0 = preprocess_img(imgf0)
        img1, tval1 = preprocess_img(imgf1)
        thr = segment_object(img0, offset=0, er=7)
        roi0 = thr*img0
        roi1 = thr*img1
        nz0 = roi0[roi0 > 0]
        nz1 = roi1[roi1 > 0]
        ave_int0 = np.median(nz0)
        ave_int1 = np.median(nz1)
        y0.append(ave_int0)
        y1.append(ave_int1)
        cmin = np.min([np.min(img0), np.min(img1)])
        cmax = np.max([np.max(img0), np.max(img1)])
        with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
            fig, ax = plt.subplots(figsize=(10,8))
            axim = ax.imshow(img0, cmap='turbo')
            # axim.set_clim(cmin, cmax)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ax.contour(thr, linewidths=0.2, colors='r')
            ax.grid(False)
            ax.axis('off')
            fig.tight_layout(pad=0)
            cb = fig.colorbar(axim, pad=0.01, format='%.4f',
                extend='both', extendrect=True, extendfrac=0.025)
            cb.outline.set_linewidth(0)
            fig.canvas.draw()
            fig.savefig(os.path.join(results_dir, 'fp0_imgs', fname0 + '.png'),
                dpi=100, bbox_inches='tight', transparent=True, pad_inches=0)
            plt.close(fig)
            fig, ax = plt.subplots(figsize=(10,8))
            axim = ax.imshow(img1, cmap='turbo')
            # axim.set_clim(cmin, cmax)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ax.contour(thr, linewidths=0.2, colors='r')
            ax.grid(False)
            ax.axis('off')
            fig.tight_layout(pad=0)
            cb = fig.colorbar(axim, pad=0.01, format='%.4f',
                extend='both', extendrect=True, extendfrac=0.025)
            cb.outline.set_linewidth(0)
            fig.canvas.draw()
            fig.savefig(os.path.join(results_dir, 'fp1_imgs', fname1 + '.png'),
                dpi=100, bbox_inches='tight', transparent=True, pad_inches=0)
            plt.close(fig)
    y = np.array(y1)/np.array(y0)
    data = [y0, y1, list(y)]
    with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
        fig, ax = plt.subplots(figsize=(12,8))
        ax = sns.violinplot(data=data, ax=ax, palette=['#1976D2', '#D32F2F', '#7B1FA2'], scale='count')
        ax.set_xticklabels(['FP0', 'FP1', 'FP1/FP0'])
        fig.savefig(os.path.join(results_dir, 'y.png'),
            dpi=200, bbox_inches='tight', transparent=False, pad_inches=0)
        plt.close(fig)
    data = np.column_stack((y0, y1, y))
    np.savetxt(os.path.join(results_dir, 'y.csv'),
        data, delimiter=',', header='fp0,fp1,fp0/fp1', comments='')
    return y


def compare_plots(root_dir, labels, fold_change=True, plot_u=True):
    with plt.style.context(('seaborn-whitegrid', custom_styles)):
        color_cycle = cycle(custom_palette[1:])
        fig, (ax0, ax) = plt.subplots(2, 1, sharex=True, figsize=(16, 10), gridspec_kw={'height_ratios': [1, 8]})
        for i, data_dir in enumerate(natsorted([x[1] for x in os.walk(root_dir)][0])):
            y_csv = os.path.join(root_dir, data_dir, 'y_combined.csv')
            y_data = pd.read_csv(y_csv)
            t = y_data['t'].values
            y_ave = y_data['y_ave']
            y_ci = y_data['y_sem']*1.96
            if plot_u:
                u_csv = os.path.join(root_dir, data_dir, 'u_combined.csv')
                u_data = pd.read_csv(u_csv)
                tu = u_data['tu_ave'].values
                u = u_data['u_ave']
            # if fold_change:
            #     y = y/np.mean(y[:5])
            color = next(color_cycle)
            label = labels[i]
            ax.fill_between(t, (y_ave - y_ci), (y_ave + y_ci), color=color, alpha=.2)
            ax.plot(y_ave, color=color, label=label)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('AU')
            if fold_change:
                ax.set_ylabel('Fold Change')
            ax.legend(loc='best')
            if plot_u:
                ax0.plot(tu, u, color='#1976D2')
                ax0.set_yticks([0, 1])
                ax0.set_ylabel('BL')
            plot_name = 'y_combined.png'
            fig.savefig(os.path.join(root_dir, plot_name),
                dpi=100, bbox_inches='tight', transparent=False)
            plt.close(fig)
        
        
def compare_fluo_imgs(img_paths, results_dir):
    setup_dirs(os.path.join(results_dir, 'imgs'))
    setup_dirs(os.path.join(results_dir, 'debug'))
    fnames = []
    ave_ints = []
    with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
        for imgf in img_paths:
            fname = os.path.splitext(os.path.basename(imgf))[0]
            fnames.append(fname)
            img, raw, bkg, den = preprocess_img(imgf)
            thr = segment_object(den, method=None, rs=100, fh=None, offset=0, er=None, cb=None)
            roi = thr*img
            nz = roi[roi > 0]
            ave_int = np.median(nz)
            ave_ints.append(ave_int)
            cmin = np.min(img)
            cmax = np.max(img)
            fig, ax = plt.subplots(figsize=(10,8))
            axim = ax.imshow(den, cmap='turbo')
            axim.set_clim(cmin, cmax)
            fontprops = font_manager.FontProperties(size=20)
            asb = AnchoredSizeBar(ax.transData, 100, u'16\u03bcm',
                color='white', size_vertical=2, fontproperties=fontprops,
                loc='lower left', pad=0.1, borderpad=0.5, sep=5, frameon=False)
            ax.add_artist(asb)
            ax.grid(False)
            ax.axis('off')
            cb = fig.colorbar(axim, pad=0.01, format='%.3f',
                extend='both', extendrect=True, extendfrac=0.03)
            cb.outline.set_linewidth(0)
            fig.tight_layout(pad=0)
            # with warnings.catch_warnings():
            #     warnings.simplefilter('ignore')
            #     ax.contour(thr, linewidths=0.3, colors='w')
            fig.canvas.draw()
            fig.savefig(os.path.join(results_dir, 'imgs', fname + '.png'),
                dpi=100, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            bg_rows = np.argsort(np.var(den, axis=1))[-100:-1:10]
            row_i = np.random.choice(bg_rows.shape[0])
            bg_row = bg_rows[row_i]
            fig, ax = plt.subplots(figsize=(10,8))
            ax.plot(raw[bg_row, :])
            ax.plot(bkg[bg_row, :])
            ax.set_title(str(bg_row))
            bg_path = os.path.join(results_dir, 'debug', '{}.png'.format(fname))
            fig.savefig(bg_path, dpi=100)
            plt.close(fig)
        fig, ax = plt.subplots(figsize=(12,8))
        for i, (ave_int, fname) in enumerate(zip(ave_ints, fnames)):
            ax.plot([i], [ave_int], marker='o')
        plt.xticks(list(range(len(fnames))), fnames)
        fig.savefig(os.path.join(results_dir, 'y.png'),
            dpi=200, bbox_inches='tight', transparent=False)
        plt.close(fig)


def compare_AUCs(root_dir, t_lim=(60, 120)):
    with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
        fig, ax = plt.subplots(figsize=(16,8))
        freqs = []
        ave_aucs = []
        color_cycle = cycle(custom_palette)
        for freq_i in natsorted([x[1] for x in os.walk(root_dir)][0]):
            freqs.append(int(freq_i))
            freq_i_dir = os.path.join(root_dir, freq_i)
            color = next(color_cycle)
            freq_i_aucs = []
            for rep_j in natsorted([x[1] for x in os.walk(freq_i_dir)][0]):
                y_csv = os.path.join(freq_i_dir, rep_j, 'y.csv')
                df = pd.read_csv(y_csv)
                span = df[(df['t'] >= t_lim[0]) & (df['t'] <= t_lim[1])]
                tspan = span['t'].values
                yspan = rescale(span['y'].values)
                # plt.plot(tspan, yspan)
                # plt.show()
                auc = simps(yspan, tspan)
                freq_i_aucs.append(auc)
                ax.plot([int(freq_i)], [auc], marker='o', color=color)
            ave_aucs.append(np.mean(freq_i_aucs))
    color = next(color_cycle)
    ax.plot(freqs, ave_aucs, color=color)
    ax.set_ylabel('AUC')
    ax.set_xlabel('Pulse Period')
    ax.set_xticks(freqs)
    fig.savefig(os.path.join(root_dir, 'freqscan.png'),
        dpi=200, bbox_inches='tight', transparent=False)
    plt.close(fig)



def barplot_expts(root_dir):
    y_data = pd.read_csv(os.path.join(root_dir, 'y.csv'))
    with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
        fig, ax = plt.subplots(figsize=(16,8))
        ax = sns.boxplot(x="System", y="Response", data=y_data, whis=np.inf)
        g = sns.stripplot(x="System", y="Response", data=y_data, ax=ax, size=10, color=".3")
        # ax.set_yscale("log")
        # g.ax.set_xticks([-0.2, 1.2])
        # plt.legend(loc='upper center', prop={"size": 20})
        ax.set_xticklabels(["6TetO", "12TetO", "6LexO", "12LexO", "6UAS", "12UAS"])
        ax.set_ylabel('Ave Fluorescence Intensity')
        ax.set_xlabel('')
        plt.savefig(os.path.join(root_dir, 'y.png'), dpi=200, transparent=False, bbox_inches='tight')
        plt.close()


def compare_groups(root_dir):
    y_data = pd.read_csv(os.path.join(root_dir, 'y.csv'))
    with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
        g = sns.catplot(x="System", y="Response", hue="Group", data=y_data,
            height=8, aspect=1.5, kind='bar', legend=False)
        g.ax.set_yscale("log")
        g.ax.set_xticks([-0.2, 1.2])
        plt.legend(loc='upper center', prop={"size": 20})
        plt.savefig(os.path.join(root_dir, 'y.png'), dpi=200, transparent=False, bbox_inches='tight')
        plt.close()


def compare_before_after(root_dir):
    y_data = pd.read_csv(os.path.join(root_dir, 'y.csv'))
    y = y_data['Response']
    palette = ['#BBDEFB', '#2196F3']
    with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(palette):
        g = sns.catplot(x="Group", y="Response", hue="Timepoint", data=y_data,
            height=8, aspect=2, kind='strip', legend=False, dodge=True, s=10)
        # g = sns.swarmplot(x="Group", y="Response", hue="Timepoint",
        #            data=y_data, height=8, aspect=1.5, dodge=True, legend=False)
        g.ax.set_xticklabels(["TetR-iLID-slow", "LexA-iLID-WT"])
        g.ax.set_xlabel('')
        g.ax.set_ylabel('Ave Fl. Intensity')
        # g.ax.set_yscale('log')
        handles, labels = g.ax.get_legend_handles_labels()
        g.ax.legend(handles, ['t=0hr', 't=24hr'], loc='best', prop={"size": 20})
        plt.savefig(os.path.join(root_dir, 'y.png'), dpi=200, transparent=False, bbox_inches='tight')
        plt.close()






# def process_cad(img_dir, results_dir):
#     """Analyze cad dataset and generate figures."""
#     setup_dirs(os.path.join(results_dir, 'imgs'))
#     t = []
#     ya = []
#     yd = []
#     imgs = []
#     tval = None
#     for i, imgf in enumerate(tqdm(list_img_files(img_dir))):
#         fname = os.path.splitext(os.path.basename(imgf))[0]
#         img, tval = preprocess_img(imgf, tval)
#         cell = segment_object(img, er=7)
#         dots = segment_clusters(img)
#         dots = np.logical_and(dots, cell)
#         anti = np.logical_xor(dots, cell)
#         croi = cell * img
#         aroi = anti * img
#         droi = dots * img
#         cnz = croi[np.nonzero(croi)]
#         anz = aroi[np.nonzero(aroi)]
#         dnz = droi[np.nonzero(droi)]
#         cell_area = len(cnz)/39.0625
#         anti_area = len(anz)/39.0625
#         dots_area = len(dnz)/39.0625
#         if i == 0:
#             cmin = np.min(img)
#             cmax = 1.1*np.max(img)
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore")
#             anti_int = np.nan_to_num(np.median(anz)) * (anti_area / cell_area)
#             dots_int = np.nan_to_num(np.median(dnz) - np.median(anz)) * (dots_area / cell_area)
#         t.append(np.float(fname))
#         ya.append(anti_int)
#         yd.append(dots_int)
#         with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
#             fig, ax = plt.subplots(figsize=(10,8))
#             axim = ax.imshow(img, cmap='turbo')
#             axim.set_clim(cmin, cmax)
#             with warnings.catch_warnings():
#                 warnings.simplefilter("ignore")
#                 ax.contour(cell, linewidths=0.4, colors='w')
#                 ax.contour(dots, linewidths=0.2, colors='r')
#             ax.grid(False)
#             ax.axis('off')
#             fig.tight_layout(pad=0)
#             cb = fig.colorbar(axim, pad=0.01, format='%.4f',
#                 extend='both', extendrect=True, extendfrac=0.025)
#             cb.outline.set_linewidth(0)
#             fig.canvas.draw()
#             imgs.append(np.array(fig.canvas.renderer._renderer))
#             fig.savefig(os.path.join(results_dir, 'imgs', fname + '.png'),
#                 dpi=100, bbox_inches='tight', transparent=True, pad_inches=0)
#             plt.close(fig)
#     data = np.column_stack((t, ya, yd))
#     np.savetxt(os.path.join(results_dir, 'y.csv'),
#         data, delimiter=',', header='t,ya,yd', comments='')
#     y = np.column_stack((ya, yd))
#     # plot(t, y, xlabel='Time (s)', ylabel='AU',
#     #     labels=['Anti Region', 'Dots Region'], save_path=os.path.join(results_dir, 'plot.png'))
#     y = np.column_stack((rescale(ya), rescale(yd)))
#     # plot(t, y, xlabel='Time (s)', ylabel='AU',
#     #     labels=['Anti Region', 'Dots Region'], save_path=os.path.join(results_dir, 'plot01.png'))
#     mimwrite(os.path.join(results_dir, 'cell.gif'), imgs, fps=len(imgs)//12)
#     return t, ya, yd


# def combine_cad(results_dir):
#     with plt.style.context(('seaborn-whitegrid', custom_styles)):
#         fig, ax = plt.subplots()
#         combined_ya = pd.DataFrame()
#         combined_yd = pd.DataFrame()
#         for i, data_dir in enumerate(natsorted([x[1] for x in os.walk(results_dir)][0])):
#             csv_fp = os.path.join(results_dir, data_dir, 'y.csv')
#             data = pd.read_csv(csv_fp)
#             t = data['t'].values
#             ya = data['ya'].values
#             yd = data['yd'].values
#             yaf = interp1d(t, ya, fill_value='extrapolate')
#             ydf = interp1d(t, yd, fill_value='extrapolate')
#             t = np.arange(round(t[0]), round(t[-1]) + 1, 1)
#             ya = pd.Series([yaf(ti) for ti in t], index=t, name=i)
#             yd = pd.Series([ydf(ti) for ti in t], index=t, name=i)
#             combined_ya = pd.concat([combined_ya, ya], axis=1)
#             combined_yd = pd.concat([combined_yd, yd], axis=1)
#             ax.plot(ya, color='#BBDEFB')
#             ax.plot(yd, color='#ffcdd2')
#         ya_ave = combined_ya.mean(axis=1).rename('ya_ave')
#         ya_std = combined_ya.std(axis=1).rename('ya_std')
#         ya_sem = combined_ya.sem(axis=1).rename('ya_sem')
#         ya_ci = 1.96*ya_sem
#         yd_ave = combined_yd.mean(axis=1).rename('yd_ave')
#         yd_std = combined_yd.std(axis=1).rename('yd_std')
#         yd_sem = combined_yd.sem(axis=1).rename('yd_sem')
#         yd_ci = 1.96*yd_sem
#         combined_data = pd.concat([ya_ave, ya_std, ya_sem, yd_ave, yd_std, yd_sem], axis=1)
#         combined_data.to_csv(os.path.join(results_dir, 'y.csv'))
#         ax.plot(ya_ave, color='#1976D2', label='Anti Region Ave')
#         ax.plot(yd_ave, color='#d32f2f', label='Dots Region Ave')
#         ax.fill_between(t, (ya_ave - ya_ci), (ya_ave + ya_ci), color='#1976D2', alpha=.1)
#         ax.fill_between(t, (yd_ave - yd_ci), (yd_ave + yd_ci), color='#d32f2f', alpha=.1)
#         ax.set_xlabel('Time (s)')
#         ax.set_ylabel('AU')
#         ax.legend(loc='best')
#         fig.savefig(os.path.join(results_dir, 'combined.png'),
#             dpi=100, bbox_inches='tight', transparent=False, pad_inches=0)
#         plt.close(fig)
 

# def linescan(img_dir, results_dir):
#     setup_dirs(os.path.join(results_dir, 'imgs'))
#     setup_dirs(os.path.join(results_dir, 'line'))
#     setup_dirs(os.path.join(results_dir, 'debug'))
#     t = []
#     y = []
#     imgs = []
#     lines = []
#     areas = []
#     for imgf in list_img_files(img_dir):
#         t.append(np.float(os.path.splitext(os.path.basename(imgf))[0]))
#     with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
#         for i, imgf in enumerate(tqdm(list_img_files(img_dir))):
#             fname = os.path.splitext(os.path.basename(imgf))[0]
#             img, raw, bkg = preprocess_img(imgf)
#             if i == 0:
#                 img = rotate(img, angle=-10)
#             if i == 1:
#                 img = rotate(img, angle=-5)
#             if i == 2:
#                 img = rotate(img, angle=-18)
#             if i == 3:
#                 img = rotate(img, angle=-35)
#             if i == 4:
#                 img = rotate(img, angle=-50)
#             if i == 5:
#                 img = rotate(img, angle=-55)
#             thr = segment_object(img, offset=0)
#             roi = thr*img
#             nz = roi[np.nonzero(roi)]
#             ave_int = np.median(nz)
#             y.append(ave_int)
#             labeled = label(thr)
#             rprops = regionprops(labeled)
#             line_row = int(np.round(rprops[0].centroid[0], 0))
#             line_col = int(np.round(rprops[0].centroid[1], 0))
#             area = rprops[0].area
#             areas.append(area)
#             if i == 0:
#                 cmin = np.min(img)
#                 cmax = 1*np.max(img)
#             fig, ax = plt.subplots(figsize=(10,8))
#             axim = ax.imshow(img, cmap='turbo')
#             ax.plot([line_col-40, line_col+40], [line_row, line_row], color='w')
#             with warnings.catch_warnings():
#                 warnings.simplefilter('ignore')
#                 ax.contour(thr, linewidths=0.3, colors='w')
#             axim.set_clim(cmin, cmax)
#             ax.grid(False)
#             ax.axis('off')
#             cb = fig.colorbar(axim, pad=0.01, format='%.3f',
#                 extend='both', extendrect=True, extendfrac=0.03)
#             cb.outline.set_linewidth(0)
#             fig.tight_layout(pad=0)
#             fig.canvas.draw()
#             fig.savefig(os.path.join(results_dir, 'imgs', fname + '.png'),
#                 dpi=100, bbox_inches='tight', pad_inches=0)
#             imgs.append(img_as_ubyte(np.array(fig.canvas.renderer._renderer)))
#             plt.close(fig)
#             bg_row = np.argmax(np.var(img, axis=1))
#             fig, ax = plt.subplots(figsize=(10,8))
#             ax.plot(raw[bg_row, :])
#             ax.plot(bkg[bg_row, :])
#             ax.set_title(str(bg_row))
#             fig.savefig(os.path.join(results_dir, 'debug', '{}.png'.format(fname)),
#                 dpi=100, bbox_inches='tight', transparent=False, pad_inches=0)
#             plt.close(fig)
#             fig, ax = plt.subplots(figsize=(12,8))
#             ax.plot(img[line_row, (line_col-40):(line_col+40)])
#             ax.set_ylabel('Pixel Intensity')
#             ax.set_xlabel('X Position')
#             ax.set_ylim(0, 1)
#             fig.savefig(os.path.join(results_dir, 'line', '{}.png'.format(fname)),
#                 dpi=100, bbox_inches='tight', transparent=False, pad_inches=0)
#             lines.append(np.array(fig.canvas.renderer._renderer))
#             plt.close(fig)
#         fig, ax = plt.subplots(figsize=(12,8))
#         ax.plot(t, y, color='#d32f2f')
#         ax.set_xlabel('Time (Frame)')
#         ax.set_ylabel('AU')
#         fig.tight_layout()
#         fig.canvas.draw()
#         fig.savefig(os.path.join(results_dir, 'median_intensity.png'),
#             dpi=300, bbox_inches='tight', transparent=False, pad_inches=0)
#         plt.close(fig)
#         fig, ax = plt.subplots(figsize=(12,8))
#         ax.plot(t, areas, color='#d32f2f')
#         ax.set_xlabel('Time (Frame)')
#         ax.set_ylabel('Area (sq. pixels)')
#         fig.tight_layout()
#         fig.canvas.draw()
#         fig.savefig(os.path.join(results_dir, 'area.png'),
#             dpi=300, bbox_inches='tight', transparent=False, pad_inches=0)
#         plt.close(fig)
#     data = np.column_stack((t, y, areas))
#     np.savetxt(os.path.join(results_dir, 'data.csv'),
#         data, delimiter=',', header='t,med_int,area', comments='')
#     with warnings.catch_warnings():
#         warnings.simplefilter('ignore')
#         mimwrite(os.path.join(results_dir, 'imgs.gif'), imgs, fps=2)
#         mimwrite(os.path.join(results_dir, 'line.gif'), lines, fps=2)
#     return t, y


# def combine_uy(root_dir, fold_change=True):
#     with plt.style.context(('seaborn-whitegrid', custom_styles)):
#         fig, (ax0, ax) = plt.subplots(
#             2, 1, sharex=True, figsize=(16, 10), gridspec_kw={'height_ratios': [1, 8]}
#         )
#         combined_y = pd.DataFrame()
#         combined_uta = pd.DataFrame()
#         combined_utb = pd.DataFrame()
#         for i, data_dir in enumerate(natsorted([x[1] for x in os.walk(root_dir)][0])):
#             y_csv_fp = os.path.join(root_dir, data_dir, 'y.csv')
#             y_data = pd.read_csv(y_csv_fp)
#             t = y_data['t'].values
#             y = y_data['y'].values
#             yf = interp1d(t, y, fill_value='extrapolate')
#             t = np.arange(round(t[0]), round(t[-1]) + 1, 1)
#             y = pd.Series([yf(ti) for ti in t], index=t, name=i)
#             if fold_change:
#                 y = y/y[0]
#             combined_y = pd.concat([combined_y, y], axis=1)
#             u_csv_fp = os.path.join(root_dir, data_dir, 'u.csv')
#             u_data = pd.read_csv(u_csv_fp)
#             uta = np.around(u_data['ta'].values, 1)
#             utb = np.around(u_data['tb'].values, 1)
#             uta = pd.Series(uta, name=i)
#             utb = pd.Series(utb, name=i)
#             combined_uta = pd.concat([combined_uta, uta], axis=1)
#             combined_utb = pd.concat([combined_utb, utb], axis=1)
#             ax.plot(y, color='#1976D2', alpha=0.5, linewidth=2)
#         y_ave = combined_y.mean(axis=1).rename('y_ave')
#         y_std = combined_y.std(axis=1).rename('y_std')
#         y_sem = combined_y.sem(axis=1).rename('y_sem')
#         data = pd.concat([y_ave, y_std, y_sem], axis=1).dropna()
#         data.to_csv(os.path.join(root_dir, 'y_combined.csv'))
#         y_ave = data['y_ave']
#         y_ci = data['y_sem']*1.96
#         uta_ave = combined_uta.mean(axis=1).rename('uta_ave')
#         utb_ave = combined_utb.mean(axis=1).rename('utb_ave')
#         tu = np.around(np.arange(t[0], t[-1], 0.1), 1)
#         u = np.zeros_like(tu)
#         for ta, tb in zip(uta, utb):
#             ia = list(tu).index(ta)
#             ib = list(tu).index(tb)
#             u[ia:ib] = 1
#         ax.fill_between(t, (y_ave - y_ci), (y_ave + y_ci), color='#1976D2', alpha=.2, label='95% CI')
#         ax.plot(y_ave, color='#1976D2', label='Ave')
#         ax.set_xlabel('Time (s)')
#         ax.set_ylabel('AU')
#         if fold_change:
#             ax.set_ylabel('Fold Change')
#         ax.legend(loc='best')
#         ax0.plot(tu, u, color='#1976D2')
#         ax0.set_yticks([0, 1])
#         ax0.set_ylabel('BL')
#         if not fold_change:
#             plot_name = 'y_combined_AU.png'
#         else:
#             plot_name = 'y_combined.png'
#         fig.savefig(os.path.join(root_dir, plot_name),
#             dpi=100, bbox_inches='tight', transparent=False)
#         plt.close(fig)