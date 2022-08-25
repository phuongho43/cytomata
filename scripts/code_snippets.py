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
        n = 0
        for i, data_dir in enumerate(natsorted([x[1] for x in os.walk(root_dir)][0])):
            n += 1
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
            ax.plot(y, color='#785EF0', alpha=1, linewidth=1)
            u_csv = os.path.join(root_dir, data_dir, 'u.csv')
            if plot_u:
                u_data = pd.read_csv(u_csv)
                tu = u_data['t'].values
                tu = pd.Series(tu, index=tu, name=i)
                u = pd.Series(u_data['u'].values, index=tu, name=i)
                combined_tu = pd.concat([combined_tu, tu], axis=1)
                combined_u = pd.concat([combined_u, u], axis=1)
        t_ave = combined_t.mean(axis=1).rename('t_ave')
        y_ave = combined_y.mean(axis=1).rename('y_ave')
        y_std = combined_y.std(axis=1).rename('y_std')
        y_sem = combined_y.sem(axis=1).rename('y_sem')
        if plot_u:
            tu_ave = combined_tu.mean(axis=1).rename('tu_ave')
            u_ave = combined_u.mean(axis=1).rename('u_ave')
            u_data = pd.concat([tu_ave, u_ave], axis=1).dropna()
            u_data.to_csv(os.path.join(root_dir, 'u.csv'), index=False)
        y_data = pd.concat([t_ave, y_ave, y_std, y_sem], axis=1).dropna()
        y_data.to_csv(os.path.join(root_dir, 'y.csv'), index=False)
        y_ave = y_data['y_ave']
        y_ci = y_data['y_sem']*1.96
        ax.fill_between(t, (y_ave - y_ci), (y_ave + y_ci), color='#785EF0', alpha=.2, label='95% CI')
        ax.plot(y_ave, color='#785EF0', label='Ave')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('AU')
        ax.text(0.96, 1.01, 'n={}'.format(n), ha='left', va='center', fontsize=16, transform=ax.transAxes)
        if fold_change:
            ax.set_ylabel('Fold Change')
        ax.legend(loc='best')
        if plot_u:
            ax0.plot(tu, u, color='#648FFF')
            ax0.set_yticks([0, 1])
            ax0.set_ylabel('BL')
        plot_name = 'y.png'
        fig.savefig(os.path.join(root_dir, plot_name),
            dpi=100, bbox_inches='tight', transparent=False)
        plt.close(fig)

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




def prep_itranslo_data(y_csv, u_csv):
    ydf = pd.read_csv(y_csv)
    udf = pd.read_csv(u_csv)
    td = np.around(ydf['t'].values, 1)
    ycd = ydf['yc'].values
    ynd = ydf['yn'].values
    t = np.around(np.arange(td[0], td[-1], 0.1), 1)
    ycf = interp1d(td, ycd)
    ynf = interp1d(td, ynd)
    yc = np.array([ycf(ti) for ti in t])
    yn = np.array([ynf(ti) for ti in t])
    uta = np.around(udf['ta'].values, 1)
    utb = np.around(udf['tb'].values, 1)
    u = np.zeros_like(t)
    for ta, tb in zip(uta, utb):
        ia = list(t).index(ta)
        ib = list(t).index(tb)
        u[ia:ib] = 1
    y = np.column_stack([yc, yn])
    return t, y, u


def fit_itranslo(t, y, u, results_dir):
    y0 = y[0, :]
    uf = interp1d(t, u, bounds_error=False, fill_value=0)
    min_res = 1e15
    best_params = None
    iter_t = time.time()
    def residual(params):
        tm, ym = sim_itranslo(t, y0, uf, params)
        res = np.mean(np.square(ym - y))
        return  res
    def opt_iter(params, iter, res):
        nonlocal min_res, best_params, iter_t
        clear_screen()
        ti = time.time()
        print('seconds/iter:', str(ti - iter_t))
        iter_t = ti
        print('Iter: {} | Res: {}'.format(iter, res))
        print(params.valuesdict())
        if res < min_res:
            min_res = res
            best_params = params.valuesdict()
        print('Best so far:')
        print('Res:', str(min_res))
        print(best_params)
    dyc = np.median(np.absolute(np.ediff1d(y[:, 0])))
    dyn = np.median(np.absolute(np.ediff1d(y[:, 1])))
    a0 = dyn/dyc
    kmax = 10**np.floor(np.log10(np.max(y)))
    dk = kmax/10
    params = lm.Parameters()
    params.add('ku', value=kmax/2, min=0, max=kmax)
    params.add('kf', value=kmax/2, min=0, max=kmax)
    params.add('kr', value=kmax/2, min=0, max=kmax)
    params.add('a', value=a0, min=1, max=10)
    ta = time.time()
    results = lm.minimize(
        residual, params, method='differential_evolution',
        iter_cb=opt_iter, nan_policy='propagate', tol=1e-1
    )
    print('Elapsed Time: ', str(time.time() - ta))
    opt_params = results.params.valuesdict()
    # opt_params = dict([('ku', 0.0008942295681229174), ('kf', 0.0005048121271898231), ('kr', 0.0006244090506147587), ('a', 4.632731164948481)])
    with open(os.path.join(results_dir, 'opt_params.json'), 'w', encoding='utf-8') as fo:
        json.dump(opt_params, fo, ensure_ascii=False, indent=4)
    tm, ym = sim_itranslo(t, y0, uf, opt_params)
    with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
        fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, figsize=(16, 10), gridspec_kw={'height_ratios': [1, 8]})
        ax0.plot(t, u)
        ax0.set_yticks([0, 1])
        ax0.set_ylabel('BL')
        ax1.plot(t, y[:, 0], color='#BBDEFB', label='Cytoplasm (Data)')
        ax1.plot(tm, ym[:, 0], color='#1976D2', label='Cytoplasm (Model)')
        ax1.plot(t, y[:, 1], color='#ffcdd2', label='Nucleus (Data)')
        ax1.plot(tm, ym[:, 1], color='#d32f2f', label='Nucleus (Model)')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('AU')
        ax1.legend(loc='best')
        fig.tight_layout()
        fig.canvas.draw()
        fig.savefig(os.path.join(results_dir, 'fit.png'),
            dpi=300, bbox_inches='tight', transparent=False, pad_inches=0)
        plt.close(fig)
    return opt_params


def prep_iexpress_data(y_csv, x_csv, u_csv):
    ydf = pd.read_csv(y_csv)
    xdf = pd.read_csv(x_csv)
    udf = pd.read_csv(u_csv)
    t = np.around(ydf['t'].values/60)
    t = np.arange(t[0], t[-1])
    y = ydf['y'].values[:-1]
    x = xdf['yn'].values[::60]
    uta = np.around(udf['ta'].values/60)
    utb = np.around(udf['tb'].values/60)
    u = np.zeros_like(t)
    ia = list(t).index(uta[0])
    ib = list(t).index(utb[-1])
    u[ia:ib] = 1
    # plt.plot(t, u)
    # plt.plot(t, x)
    # plt.plot(t, y)
    # plt.show()
    return t, y, x, u


def fit_iexpress(t, y, x, u, results_dir):
    y0 = [0, y[0]]
    xf = interp1d(t, x, bounds_error=False, fill_value='extrapolate')
    min_res = 1e15
    best_params = None
    iter_t = time.time()
    # def residual(params):
    #     tm, ym = sim_iexpress(t, y0, xf, params)
    #     res = np.mean((ym[:, 1] - y)**2)
    #     return  res
    # def opt_iter(params, iter, res):
    #     nonlocal min_res, best_params, iter_t
    #     clear_screen()
    #     ti = time.time()
    #     print('seconds/iter:', str(ti - iter_t))
    #     iter_t = ti
    #     print('Iter: {} | Res: {}'.format(iter, res))
    #     print(params.valuesdict())
    #     if res < min_res:
    #         min_res = res
    #         best_params = params.valuesdict()
    #     print('Best so far:')
    #     print('Res:', str(min_res))
    #     print(best_params)
    # ta = time.time()
    # # params = lm.Parameters()
    # # params.add('ka', min=0.01, max=1, brute_step=0.01)
    # # params.add('kb', min=0.1, max=10, brute_step=0.1)
    # # params.add('kc', min=0.001, max=0.1, brute_step=0.001)
    # # params.add('n', min=1, vary=False)
    # # results = lm.minimize(
    # #     residual, params, method='brute',
    # #     iter_cb=opt_iter, nan_policy='propagate',
    # # )
    # params = lm.Parameters()
    # params.add('ka', value=0.1, min=0, max=1)
    # params.add('kb', value=0.1, min=0, max=1)
    # params.add('kc', value=0.1, min=0, max=1)
    # params.add('n', value=1, min=0, max=1)
    # params.add('kf', value=0.1, min=0, max=1)
    # params.add('kg', value=0.1, min=0, max=1)
    # ta = time.time()
    # results = lm.minimize(
    #     residual, params, method='dual_annealing',
    #     iter_cb=opt_iter, nan_policy='propagate', tol=1e-3
    # )
    # print('Elapsed Time: ', str(time.time() - ta))
    # opt_params = params
    # opt_params = results.params.valuesdict()
    # with open(os.path.join(results_dir, 'opt_params.json'), 'w', encoding='utf-8') as fo:
    #     json.dump(opt_params, fo, ensure_ascii=False, indent=4)
    opt_params = dict([('ka', 0.10014001562994905), ('kb', 0.17638515989679424), ('kc', 0.002468692274783906), ('n', 0.9758946268030773), ('kf', 0.0008406127261191349), ('kg', 0.6462795804896208)])
    tm, ym = sim_iexpress(t, y0, xf, opt_params)
    with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
        fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, figsize=(16, 10), gridspec_kw={'height_ratios': [1, 8]})
        ax0.plot(t, u)
        ax0.set_yticks([0, 1])
        ax0.set_ylabel('BL')
        ax1.scatter(t, y, color='#ffcdd2', label='POI (Data)')
        # ax1.plot(tm, ym[:, 0], color='#2196F3', label='mRNA (Model)')
        ax1.plot(tm, ym[:, 1], color='#d32f2f', label='POI (Model)')
        ax1.set_xlabel('Time (m)')
        ax1.set_ylabel('AU')
        ax1.legend(loc='best')
        fig.tight_layout()
        fig.canvas.draw()
        fig.savefig(os.path.join(results_dir, 'fit.png'),
            dpi=300, bbox_inches='tight', transparent=False, pad_inches=0)
        plt.close(fig)
    return opt_params


    
    # y_csv = '/home/phuong/data/LINTAD/LINuS-results/0/y.csv'
    # u_csv = '/home/phuong/data/LINTAD/LINuS/u0.csv'
    # res_dir = '/home/phuong/data/LINTAD/LINuS-results/0/'
    # t, y, u = prep_itranslo_data(y_csv, u_csv)
    # fit_itranslo(t, y, u, res_dir)


    # root_dir = '/home/phuong/data/LINTAD/LINuS-mock/'
    # u_csv = '/home/phuong/data/LINTAD/LINuS/u0.csv'
    # for data_dirname in natsorted([x[1] for x in os.walk(root_dir)][0]):
    #     y_csv = os.path.join(root_dir, data_dirname, 'y.csv')
    #     res_dir = os.path.join(root_dir, data_dirname)
    #     t, y, u = prep_itranslo_data(y_csv, u_csv)
    #     fit_itranslo(t, y, u, res_dir)
    # ku_vals = []
    # kf_vals = []
    # kr_vals = []
    # a_vals = []
    # for data_dirname in natsorted([x[1] for x in os.walk(root_dir)][0]):
    #     params_path = os.path.join(root_dir, data_dirname, 'opt_params.json')
    #     with open(params_path) as f:
    #         params = json.load(f)
    #         ku_vals.append(params['ku'])
    #         kf_vals.append(params['kf'])
    #         kr_vals.append(params['kr'])
    #         a_vals.append(params['a'])
    # data = [ku_vals, kf_vals, kr_vals]
    # with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
    #     fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 10), gridspec_kw={'width_ratios': [6, 2]})
    #     ax0 = sns.violinplot(data=data, ax=ax0, palette=['#1976D2', '#D32F2F', '#388E3C'])
    #     ax0.set_xticklabels(['ku', 'kf', 'kr'])
    #     ax1 = sns.violinplot(data=a_vals, ax=ax1, color='#F57C00')
    #     ax1.set_xticklabels(['a'])
    #     fig.savefig(os.path.join(root_dir, 'params_dist.png'),
    #         dpi=200, bbox_inches='tight', transparent=False, pad_inches=0)
    #     plt.close(fig)


    # root_dir = '/home/phuong/data/ILID/RA-16I/20200921-B3-sspBu_RA-16I_spike/results/'
    # u_csv = os.path.join(root_dir, 'u_combined.csv')
    # y_csv = os.path.join(root_dir, 'y_combined.csv')
    # u_data = pd.read_csv(u_csv)
    # tu = u_data['tu_ave'].values
    # u = u_data['u_ave'].values
    # uf = interp1d(tu, u, bounds_error=False, fill_value=0)
    # y_data = pd.read_csv(y_csv)
    # t = y_data['t'].values
    # y = y_data['y_ave'].values
    # # y = y - y[0]
    # yf = interp1d(t, y, fill_value='extrapolate')
    # y = np.array([yf(ti) for ti in tu])
    # fit_ilid(tu, y, u, root_dir)

    # t = np.arange(0, 600)
    # y0 = [y[0], 0]
    # u = np.zeros_like(t)
    # # p = 20
    # # w = 1
    # # for i in range(t[60], t[480], p):
    # #     u[i:i+w] = 1
    # u[60:480] = 1
    # params = {
    #     "ka": 1.029336557988737,
    #     "kb": 1.2472229538949358,
    #     "kc": 0.34861647650637984,
    #     "n": 1.0961954940397614
    # }
    # uf = interp1d(t, u, bounds_error=False, fill_value=0)
    # tm, ym = sim_ilid(t, y0, uf, params)
    # with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
    #     fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, figsize=(16, 10), gridspec_kw={'height_ratios': [1, 8]})
    #     ax0.plot(t, u)
    #     ax0.set_yticks([0, 1])
    #     ax0.set_ylabel('BL')
    #     ax1.plot(tm, ym[:, 0], color='#1976D2', label='A')
    #     ax1.plot(tm, ym[:, 1], color='#d32f2f', label='B')
    #     ax1.set_xlabel('Time (s)')
    #     ax1.set_ylabel('AU')
    #     ax1.legend(loc='best')
    #     fig.tight_layout()
    #     fig.canvas.draw()
    #     save_dir = '/home/phuong/data/ILID/RA-HF/20200921-B3-sspBu_RA-27V_spike/results/'
    #     fig.savefig(os.path.join(save_dir, 'sim.png'),
    #         dpi=300, bbox_inches='tight', transparent=False)
    #     plt.close(fig)


    # y0 = [0.1, 0, 0.5, 0, 0.05]
    # # uf = interp1d(tu, u, bounds_error=False, fill_value=0)
    # t = np.arange(0, 600)
    # u = np.zeros_like(t)
    # p = 20
    # w = 1
    # for i in range(t[60], t[540], p):
    #     u[i:i+w] = 1
    # uf = interp1d(t, u, bounds_error=False, fill_value=0)
    # params = {
    #     'k1f': 0.23593737155206962,
    #     'k1r': 0.005057247900003281,
    #     'k2f': 0.5858908062641166,
    #     'k2r': 0.3465577497760164,
    # }
    # tm, ym = sim_fresca(t, y0, uf, params)
    # with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
    #     fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, figsize=(16, 10), gridspec_kw={'height_ratios': [1, 8]})
    #     ax0.plot(t, u)
    #     ax0.set_yticks([0, 1])
    #     ax0.set_ylabel('BL')
    #     ax1.plot(tm, ym[:, 1], color='#1976D2', label='iLID_slow')
    #     ax1.plot(tm, ym[:, 3], color='#d32f2f', label='iLID_fast')
    #     ax1.plot(tm, ym[:, 4], color='#388E3C', label='sspB')
    #     ax1.set_xlabel('Time (s)')
    #     ax1.set_ylabel('AU')
    #     ax1.legend(loc='best')
    #     fig.tight_layout()
    #     fig.canvas.draw()
    #     save_dir = '/home/phuong/data/ILID/FResCA/'
    #     fig.savefig(os.path.join(save_dir, 'sim1.png'),
    #         dpi=300, bbox_inches='tight', transparent=False)
    #     plt.close(fig)


    # y_csv = '/home/phuong/data/ILID/RA-HF/20200804-RA-HF/results/5/y.csv'
    # u_csv = '/home/phuong/data/ILID/RA-HF/20200804-RA-HF/results/5/u.csv'
    # res_dir = '/home/phuong/data/ILID/RA-HF/20200804-RA-HF/results/5/'
    # u_data = pd.read_csv(u_csv)
    # tu = u_data['t'].values
    # u = u_data['u'].values
    # y_data = pd.read_csv(y_csv)
    # t = y_data['t'].values
    # y = y_data['y'].values
    # yf = interp1d(t, y, fill_value='extrapolate')
    # y = np.array([yf(ti) for ti in tu])
    # # ya = (-y + 2*np.min(y) + (np.max(y)-np.min(y)))
    # fit_CaM_M13(tu, y, u, res_dir)


    # y_csv = '/home/phuong/data/LINTAD/LexA-results/0/y.csv'
    # x_csv = '/home/phuong/data/LINTAD/TF/y.csv'
    # u_csv = '/home/phuong/data/LINTAD/LexA/u0.csv'
    # res_dir = '/home/phuong/data/LINTAD/LexA-results/0/'
    # t, y, x, u = prep_iexpress_data(y_csv, x_csv, u_csv)
    # x = np.zeros_like(t)
    # x[130:] = 1
    # fit_iexpress(t, y, x, u, res_dir)
    
    # xf = interp1d(t, x, bounds_error=False, fill_value='extrapolate')
    # y0 = [0, y[0]]
    # params = {'ka': 0.122, 'kb': 0.1, 'kc': 0.2, 'n': 1, 'kf': 0.4, 'kg': 0.1}
    # ta = time.time()
    # tm, ym = sim_iexpress(t, y0, xf, params)
    # print(time.time() - ta)
    # plt.plot(t, y)
    # plt.plot(tm, ym[:, 0])
    # plt.show()

    # t = np.arange(0, 100, 1)
    # omega = 20
    # tau = 32
    # n = 4
    # C = []
    # for ti in t:
    #     if ti >= 32 and ti < 35:
    #         Ci = 0.1 + 0.9*np.sin(omega*(ti - tau))**n
    #     else:
    #         Ci = 0.1
    #     C.append(Ci)
    # # plt.plot(t, C)
    # # plt.show()
    # C0 = np.ones_like(t) * 0.1
    # Cf = interp1d(t, C0, bounds_error=False, fill_value=0.1)
    # y0 = [0, 0, 0, 0, 0]
    # t, y = sim_CaM_M13(t, y0, Cf)
    # y0 = y[-1, :]
    # Cf = interp1d(t, C, bounds_error=False, fill_value=0.1)
    # t, y = sim_CaM_M13(t, y0, Cf)
    # Pb = y[:, 2] + y[:, 3] + y[:, 4]
    # # plt.plot(t, C)
    # plt.plot(t, y[:, 3])
    # plt.show()


def process_timelapse(img_dir, save_dir,
    t_unit='s', sb_microns=None, cmax=None, segmt_factor=1, adj_bright=False):
    """Analyze fluorescence timelapse images and generate figures."""
    # setup_dirs(os.path.join(save_dir, 'tracks'))
    setup_dirs(os.path.join(save_dir, 'subtracted'))
    factor = segmt_factor
    t = [float(os.path.splitext(os.path.basename(imgf))[0]) for imgf in list_img_files(img_dir)]
    imgs = []
    data = []
    for i, imgf in enumerate(tqdm(list_img_files(img_dir))):
        fname = os.path.splitext(os.path.basename(imgf))[0]
        ti = float(fname)
        img, raw, bkg, den = preprocess_img(imgf)
        cmax_i = cmax
        if cmax is None:
            cmax_i = np.percentile(img, 99.99)
        plot_bkg_profile(fname, save_dir, raw, bkg)
        if adj_bright:
            a_reg = img[img > 0]
            if i == 0:
                kval = np.mean(a_reg[a_reg > np.percentile(a_reg, 95)])
            segmt_factor = factor * (kval/np.mean(a_reg[a_reg > np.percentile(a_reg, 95)]))
        thr, reg, n = segment_object(den, factor=segmt_factor)
        data_row = {'id': id, 'time': ti, 'mean_int': mi, 'b_box': bb}
            data.append(data_row)
        img_path = os.path.join(save_dir, 'subtracted', fname + '.tiff')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imsave(img_path, img_as_uint(rescale_intensity(img)))
        img_save_dir = os.path.join(save_dir, 'denoised')
        cell_den = plot_cell_img(den, None, fname, img_save_dir,
                cmax=cmax_i, t_unit=t_unit, sb_microns=sb_microns)
        img_save_dir = os.path.join(save_dir, 'outlined')
        cell_den = plot_cell_img(den, thr, fname, img_save_dir,
                cmax=cmax_i, t_unit=t_unit, sb_microns=sb_microns)
        imgs.append(cell_den)
    df = pd.DataFrame(data, index=t, columns=['id', 'time', 'mean_int', 'b_box'])
    df.dropna(how='all', inplace=True)
    df.to_csv(os.path.join(save_dir, 'y.csv'), index=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mimwrite(os.path.join(save_dir, 'cell.gif'), imgs, fps=len(imgs)//10)


def plot_sc_tracks(root_dir, min_trk_len=150, figsize=(16, 8)):
    df = pd.read_csv(os.path.join(root_dir, 'y.csv'))
    df.dropna(how='any', inplace=True)
    df.reset_index(drop=True, inplace=True)
    value_counts = df['id'].value_counts()
    keep = value_counts[value_counts >= min_trk_len].index.tolist()
    df = df[df['id'].isin(keep)]
    with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
        fig, ax = plt.subplots(figsize=figsize)
        # for idi in df['id'].unique():
        #     df_i = df.loc[(df['id'] == idi), ('time', 'mean_int')]
        sns.lineplot(data=df[['id', 'time', 'mean_int']], x='time', y='mean_int', hue='id', lw=2, palette=sns.color_palette("Blues", as_cmap=True))
        ax.set_xlabel('Time')
        ax.set_ylabel('AU')
        ax.get_legend().remove()
        fig_name = 'tracks.png'
        plt.savefig(os.path.join(save_dir, fig_name), dpi=200, transparent=False, bbox_inches='tight')
        plt.close()