�}� 	 2  ^sߥ0}J$��:!�ޢ4�!�)kݼ��ic(��� �Oo|��HrV�<��[Xkj+�36|?�U6�Ce�>��6P.�uߟ]�6��<];�r`#!��/*��و^�if�ߡ���2�n��@q�b�JK����r
<G���,�<�gqk�NpX��E����H?mva�> N�UZgK��?+ہ9��A��?�A՟����K�j��R�h������xj��2��a�ݼ�_N�y��r΍��vH��-{�}��˅�7Ak���G���S~��R�WKY3�Kq�N�R���A1ZvW�d�m!?��� ����lK��Td]č1ܣ�Kv�F��YwfX�_ս��b=/_� U��X��eDj�v�Ogi3c�Sf���nD��
��S�h��4���F��I|�'o���Y�a�w�o��B��{��Iܠ�O�鞰�yG<���/�H�}�,�!Ca���}�<-��Ly������0YX��Y�g���ؙ�'lDC���R��{Ǝj��#uea�Z�Ѧ�)��$`	bŬ�En�٦����fC,1ݾ �P�^�v(9��ߑO���ش�7w]�{d(?��ٷt�=y�J�1����&F�L�ZF�"��[Iڑ�5~owW�s�z�+K� ��)��?W�]�g��ةe�ޝ<ĴnL�u�V�,'�u��<:�V� ��{aM,���s���>&,�	G}����k�-G�P�N	��j������8�8��涊���<1õ�?�]@G��L�<�_�ę2����iZ$>u;ө�)ȃ5���Mp3�x)�tK𛾍����VI9 ~;@Ƒ�6l������-�k�}9@�4�x��29?�׿�Ze��� �]�����4�][��m�$�S�͹O$�l���t�����y�7�NP�lo"n���9�\Qcqx��.��ա��EQ�)<�X�Z-���A�0^q��/���SR���C��ʧԠ����W�/E;.m�IlD!!���}(��^�s@��j��Ԓ��V��a����?�B=	-�� �y��f���|���2�Auq�x�3�=��M���νtk0lrH+^[%���w�ƶ9[z�D}��o�!H�P;vH�2�I�Ϙ@�o���T2��|����*D˶sI~�n����D��2���"S �S7q���M��#ؿ"����"K37[2Eq�E�D���)0I͡m�)�W²0t!��ѧ�<�lj��阯��X&-���q�$�o'�?TI�q�� /��D�*R�愷��Q�@��/O�rH��u�ѷ�4߬ }(0Zڍd�d��@�ʣ"���O?;�l� �G��^!� ��z9�mUF���%%�����)/�u�!"�`�.�L{��OCD����P�ƍ�ji����g�<ԗ4m8�sWv�3>�g1Б�3�������Kgi�D�-�aR
W�R0P`��3�F�˕�^�َ��J	���R-�P�3ܻZ"O�t	���*c��4��^�xSQ�����[s+�l6�S� � �S/*��S����yv�ވ��S�ŝ�zd�~�^R�����=�;�pŋ��Z�q����"�t7�l�kI;���D� 6
[K���W�J���7Y�?a
L%9y]�5^��{R���>e���VȭY2�y�*m�I	��ƚY����%�E�Ќzw	S%�TU�:�5>�MF�`��_!��n�F��ˍN��d�x��`������v�ʓ�5]a�"������#b&������B��6T5��L��ύ̥����B�����@R4�,�E�L�G~�M��{9m�Et�F�z��쓇�VS�-,��Z�0�:��u�i=:MΓ����]��	�X
�0�z�V���"z6���Ơ7�r�@n������mm�6�{[��}O��ۇ�vb�z���p5Um�G+H�ڝ")fԕ��>�ѻ�O̶y:�3�1A�Q�:$v�����O���?GZ@�XM�����4�-4��$j#��3�i�e�h7�`�t�Ѡ��#Z���� l+z�O&���'�%G���Z��)*�ӯA�.�r�M�Ѷ�$�X8A�*��,=t�����e[�{�~0R����||�}x.�'�n'��@)e��z�;9�C�@~L�,MH0�n@t�2��� �ѓ���1a��Mw�,(�R�Z�{p�R�?��R�����K�A|�H��d�p���PD�Rf ;�XT��o\o ���X)M�� �$,���y�4�ݜ�>�V��!�ʂW�(����:�t�M�F!�b��W��n����;m�h>�C�5�ag\�Vt&��/;~��Čfs��&S�ds���.i�8��y�CFr�J/�p;� !�%��<CE�	�M� �o��9s�ʛ�0چR�xh_�B�%�8sJ쥽Y�����E�ݗ����.������[��E�M���!�VHt��]s���9��|s�����j������������ܼ�aNb�n"��n������Tp��V�ӈwI^���[�K��kF��5lt.show()

ash_cluster(data_cluster)

k_dict = dict(zip(k_cluster, k_score))
k_best = max(k_dict, key=lambda x: k_dict[x])

# 聚类框架
model_cluster = KMeans(n_clusters=k_best, random_state=9)
data_cluster_pred = pd.DataFrame(model_cluster.fit_predict(data_cluster), columns=['category'])
data_cluster_merge = pd.concat([data, data_cluster_pred], axis=1)


count_cluster = pd.crosstab(data_cluster_merge['入洗配比'], data_cluster_merge['category'])
count_cluster.plot(kind = 'bar')

